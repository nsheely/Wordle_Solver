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
}
