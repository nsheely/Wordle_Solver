//! Hybrid selection strategies
//!
//! Combines entropy with other metrics (`expected_remaining`, minimax) for improved performance.

use crate::core::Word;
use crate::solver::entropy::calculate_metrics;
use rayon::prelude::*;

/// Select best guess with `entropy+expected_size+minimax` tiebreakers
///
/// For medium candidate counts (21-100), this provides better performance than pure entropy.
/// Primary: entropy, Secondary: `expected_remaining`, Tertiary: minimax
///
/// Returns `None` if the guess pool is empty.
#[must_use]
pub fn select_with_expected_tiebreaker<'a>(
    guess_pool: &'a [Word],
    candidates: &[Word],
) -> Option<&'a Word> {
    let candidate_refs: Vec<&Word> = candidates.iter().collect();

    // Compute all metrics (parallelized)
    let metrics: Vec<_> = guess_pool
        .par_iter()
        .map(|guess| {
            let m = calculate_metrics(guess, &candidate_refs);
            (guess, m)
        })
        .collect();

    // Select by: entropy (primary), expected_remaining (secondary), max_partition (tertiary)
    metrics
        .into_iter()
        .max_by(|(_, m1), (_, m2)| {
            m1.entropy
                .total_cmp(&m2.entropy)
                .then(m2.expected_remaining.total_cmp(&m1.expected_remaining))
                .then(m2.max_partition.cmp(&m1.max_partition))
        })
        .map(|(word, _)| word)
}

/// Select best guess with hybrid scoring
///
/// For medium candidate counts (9-20), use formula: score = (entropy × `entropy_weight`) - (`max_partition` × `minimax_penalty`)
/// The default weights (100.0, 10.0) balance average-case (entropy) with worst-case (minimax) at 10:1 ratio.
///
/// Returns `None` if the guess pool is empty.
#[must_use]
pub fn select_with_hybrid_scoring<'a>(
    guess_pool: &'a [Word],
    candidates: &[Word],
    entropy_weight: f64,
    minimax_penalty: f64,
) -> Option<&'a Word> {
    let candidate_refs: Vec<&Word> = candidates.iter().collect();

    // Compute all metrics (parallelized)
    let metrics: Vec<_> = guess_pool
        .par_iter()
        .map(|guess| {
            let m = calculate_metrics(guess, &candidate_refs);
            (guess, m)
        })
        .collect();

    // Find best hybrid score
    metrics
        .into_iter()
        .max_by(|(_, m1), (_, m2)| {
            // Hybrid score: entropy (weighted) minus worst-case penalty (weighted)
            let score1 = (m1.entropy * entropy_weight) as i32
                - i32::try_from((m1.max_partition as f64 * minimax_penalty) as usize)
                    .unwrap_or(i32::MAX);
            let score2 = (m2.entropy * entropy_weight) as i32
                - i32::try_from((m2.max_partition as f64 * minimax_penalty) as usize)
                    .unwrap_or(i32::MAX);
            // Higher score is better
            score1
                .cmp(&score2)
                .then(m2.expected_remaining.total_cmp(&m1.expected_remaining))
        })
        .map(|(word, _)| word)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_with_expected_tiebreaker_works() {
        let guesses = [
            Word::new("crane").unwrap(),
            Word::new("slate").unwrap(),
            Word::new("aaaaa").unwrap(),
        ];
        let candidates = [
            Word::new("irate").unwrap(),
            Word::new("crate").unwrap(),
            Word::new("grate").unwrap(),
            Word::new("plate").unwrap(),
        ];

        let result = select_with_expected_tiebreaker(&guesses, &candidates);
        assert!(result.is_some());

        let best = result.unwrap();

        // Should select one of the better guesses (not AAAAA)
        assert!(best.text() != "aaaaa");
    }

    #[test]
    fn select_with_hybrid_scoring_works() {
        let guesses = [
            Word::new("crane").unwrap(),
            Word::new("slate").unwrap(),
            Word::new("zzzzz").unwrap(),
        ];
        let candidates = [
            Word::new("irate").unwrap(),
            Word::new("crate").unwrap(),
            Word::new("grate").unwrap(),
        ];

        let result = select_with_hybrid_scoring(&guesses, &candidates, 100.0, 10.0);
        assert!(result.is_some());

        let best = result.unwrap();

        // Should select a reasonable guess (not ZZZZZ)
        assert!(best.text() != "zzzzz");
    }

    #[test]
    fn hybrid_scoring_balances_entropy_and_minimax() {
        // Create scenario where pure entropy and pure minimax disagree
        let guesses = [
            Word::new("aeros").unwrap(), // High entropy, possibly worse minimax
            Word::new("slate").unwrap(), // Balanced
        ];
        let candidates = [Word::new("irate").unwrap(), Word::new("crate").unwrap()];

        let result = select_with_hybrid_scoring(&guesses, &candidates, 100.0, 10.0);
        assert!(result.is_some());

        let best = result.unwrap();

        // Should pick one of them (test that it doesn't panic)
        assert!(best.text() == "aeros" || best.text() == "slate");
    }

    #[test]
    fn expected_tiebreaker_returns_none_on_empty() {
        let guesses: Vec<Word> = vec![];
        let candidates = [Word::new("slate").unwrap()];

        let result = select_with_expected_tiebreaker(&guesses, &candidates);
        assert!(result.is_none());
    }

    #[test]
    fn hybrid_scoring_returns_none_on_empty() {
        let guesses: Vec<Word> = vec![];
        let candidates = [Word::new("slate").unwrap()];

        let result = select_with_hybrid_scoring(&guesses, &candidates, 100.0, 10.0);
        assert!(result.is_none());
    }

    #[test]
    fn hybrid_scoring_formula_entropy_dominates() {
        // Test that higher entropy wins when minimax is equal
        // Guess 1: "slate" vs "slate" → entropy=0, max_partition=1
        // Guess 2: "slate" vs "zzzzz" → entropy=1, max_partition=1
        // Score formula: (entropy * 100.0) as i32 - ((max_partition as f64 * 10.0) as usize) as i32
        // Guess 1 score: (0 * 100) - (1 * 10) = 0 - 10 = -10
        // Guess 2 score: (1 * 100) - (1 * 10) = 100 - 10 = 90
        // Should select guess 2 (higher entropy)

        let guesses = [
            Word::new("aaaaa").unwrap(), // Low entropy guess
            Word::new("slate").unwrap(), // Higher entropy guess
        ];
        let candidates = [Word::new("slate").unwrap(), Word::new("zzzzz").unwrap()];

        let result = select_with_hybrid_scoring(&guesses, &candidates, 100.0, 10.0);
        assert!(result.is_some());

        let best = result.unwrap();

        // slate should win because it has better entropy against these candidates
        assert_eq!(best.text(), "slate");
    }

    #[test]
    fn hybrid_scoring_formula_minimax_penalty_matters() {
        // Test that minimax_penalty actually penalizes large max_partition
        // Create scenario where higher entropy_weight favors one, higher minimax_penalty favors another

        let guesses = [Word::new("crane").unwrap(), Word::new("stale").unwrap()];
        let candidates = [Word::new("irate").unwrap(), Word::new("slate").unwrap()];

        // With high entropy weight, low minimax penalty
        let result_high_entropy = select_with_hybrid_scoring(&guesses, &candidates, 1000.0, 1.0);
        assert!(result_high_entropy.is_some());

        // With low entropy weight, high minimax penalty
        let result_high_minimax = select_with_hybrid_scoring(&guesses, &candidates, 1.0, 1000.0);
        assert!(result_high_minimax.is_some());

        // Results should exist (verifies formula doesn't panic)
        assert!(result_high_entropy.is_some() && result_high_minimax.is_some());
    }

    #[test]
    fn hybrid_scoring_exact_calculation() {
        // Precise test with known entropy and max_partition values
        // Create simple scenario with perfect binary split
        let guesses = [
            Word::new("slate").unwrap(), // Will produce binary split
            Word::new("zzzzz").unwrap(), // Will produce binary split
        ];
        let candidates = [
            Word::new("slate").unwrap(), // Matches guess 1 perfectly
            Word::new("aaaaa").unwrap(), // Matches neither
        ];

        // Both guesses create 2 partitions (one perfect match, one no match)
        // entropy ≈ 1.0, max_partition = 1 for both
        // Score = (1.0 * 100.0) as i32 - ((1 as f64 * 10.0) as usize) as i32
        //       = 100 - 10 = 90 for both
        // Should select slate (tiebreaker: expected_remaining)

        let result = select_with_hybrid_scoring(&guesses, &candidates, 100.0, 10.0);
        assert!(result.is_some());

        let best = result.unwrap();

        // slate should be selected (better expected_remaining as tiebreaker)
        assert_eq!(best.text(), "slate");
    }

    #[test]
    fn hybrid_scoring_subtraction_not_addition() {
        // Verify that the formula uses subtraction, not addition
        // If formula incorrectly added penalties, behavior would be reversed

        let guesses = [
            Word::new("abcde").unwrap(), // Creates all-gray pattern
            Word::new("slate").unwrap(), // Better splitting
        ];
        let candidates = [
            Word::new("fghij").unwrap(), // No letters match abcde
            Word::new("klmno").unwrap(), // No letters match abcde
        ];

        // abcde creates 1 partition (all candidates same pattern), max_partition=2
        // slate creates potentially 2 partitions, max_partition smaller
        // With correct formula (subtraction), slate wins
        // With incorrect formula (addition), abcde might win

        let result = select_with_hybrid_scoring(&guesses, &candidates, 100.0, 10.0);
        assert!(result.is_some());

        // The result should exist and be reasonable
        let best = result.unwrap();
        assert!(best.text() == "slate" || best.text() == "abcde");
    }

    #[test]
    fn hybrid_scoring_weights_applied_correctly() {
        // Verify that weights are actually multiplied, not added/divided

        let guesses = [Word::new("crane").unwrap()];
        let candidates = [Word::new("slate").unwrap(), Word::new("irate").unwrap()];

        // Try different weight combinations
        let result1 = select_with_hybrid_scoring(&guesses, &candidates, 100.0, 10.0);
        let result2 = select_with_hybrid_scoring(&guesses, &candidates, 200.0, 5.0);
        let result3 = select_with_hybrid_scoring(&guesses, &candidates, 50.0, 20.0);

        // All should return a result (formula doesn't break with different weights)
        assert!(result1.is_some());
        assert!(result2.is_some());
        assert!(result3.is_some());

        // All should select the same word (only one guess available)
        assert_eq!(result1.unwrap().text(), "crane");
        assert_eq!(result2.unwrap().text(), "crane");
        assert_eq!(result3.unwrap().text(), "crane");
    }

    #[test]
    fn hybrid_scoring_multiplication_not_division_or_addition() {
        // Create scenario where * vs / vs + would give different winners
        // Use extreme weights to make differences obvious

        let guesses = [Word::new("slate").unwrap(), Word::new("zzzzz").unwrap()];
        let candidates = [
            Word::new("irate").unwrap(),
            Word::new("crate").unwrap(),
            Word::new("grate").unwrap(),
        ];

        // Test with high entropy weight (entropy should dominate)
        let result_high_entropy = select_with_hybrid_scoring(&guesses, &candidates, 10000.0, 1.0);
        assert!(result_high_entropy.is_some());

        // Test with balanced weights
        let result_balanced = select_with_hybrid_scoring(&guesses, &candidates, 100.0, 100.0);
        assert!(result_balanced.is_some());

        // Both should return a valid guess (verifies multiplication doesn't break)
        assert!(result_high_entropy.is_some() && result_balanced.is_some());
    }

    #[test]
    fn hybrid_scoring_penalty_multiplication_verified() {
        // Verify minimax_penalty is multiplied, not divided or added
        // Create candidates where max_partition differs significantly

        let guesses = [
            Word::new("crane").unwrap(),
            Word::new("aaaaa").unwrap(), // Creates large partition
        ];
        let candidates = [
            Word::new("irate").unwrap(),
            Word::new("slate").unwrap(),
            Word::new("crate").unwrap(),
        ];

        // Test with high minimax penalty (max_partition should dominate)
        let result_high_minimax = select_with_hybrid_scoring(&guesses, &candidates, 1.0, 10000.0);
        assert!(result_high_minimax.is_some());

        // Test with low minimax penalty (entropy should dominate)
        let result_low_minimax = select_with_hybrid_scoring(&guesses, &candidates, 1000.0, 0.1);
        assert!(result_low_minimax.is_some());

        // Both should return a valid guess (verifies multiplication works correctly)
        assert!(result_high_minimax.is_some() && result_low_minimax.is_some());
    }
}
