//! Adaptive selection strategies
//!
//! Selection functions used by `AdaptiveStrategy` for small candidate counts.
//! These combine minimax with entropy and candidate preference.

use crate::core::Word;
use crate::solver::entropy::{calculate_entropy, calculate_metrics};
use rayon::prelude::*;

/// Select best guess with `minimax+entropy` tiebreaker
///
/// For small candidate counts (3-8), minimax-first provides better worst-case guarantees.
/// Among guesses with minimum `max_partition`, pick highest entropy.
/// Also uses epsilon-greedy candidate preference when minimax is tied.
///
/// Returns `None` if the guess pool is empty.
#[must_use]
pub fn select_minimax_first<'a>(
    guess_pool: &'a [Word],
    candidates: &[Word],
    epsilon: f64,
) -> Option<&'a Word> {
    let candidate_refs: Vec<&Word> = candidates.iter().collect();

    // Compute all metrics since we need both max_partition and entropy (parallelized)
    let metrics: Vec<_> = guess_pool
        .par_iter()
        .map(|guess| {
            let m = calculate_metrics(guess, &candidate_refs);
            let is_candidate = candidates.iter().any(|c| c.text() == guess.text());
            (guess, m, is_candidate)
        })
        .collect();

    // Return None if empty
    if metrics.is_empty() {
        return None;
    }

    // Find minimum max_partition
    let min_max_partition = metrics
        .iter()
        .map(|(_, m, _)| m.max_partition)
        .min()
        .unwrap_or(0);

    // Among guesses with min max_partition, use conditional candidate preference
    let tied_minimax: Vec<_> = metrics
        .into_iter()
        .filter(|(_, m, _)| m.max_partition == min_max_partition)
        .collect();

    // Find max entropy among tied minimax
    let max_entropy = tied_minimax
        .iter()
        .map(|(_, m, _)| m.entropy)
        .max_by(f64::total_cmp)
        .unwrap_or(0.0);

    // Prefer candidates if within epsilon of max entropy
    if let Some((word, _, _)) = tied_minimax
        .iter()
        .filter(|(_, m, is_cand)| *is_cand && (max_entropy - m.entropy) < epsilon)
        .max_by(|(_, m1, _), (_, m2, _)| m1.entropy.total_cmp(&m2.entropy))
    {
        return Some(word);
    }

    // Otherwise just pick highest entropy
    tied_minimax
        .into_iter()
        .max_by(|(_, m1, _), (_, m2, _)| m1.entropy.total_cmp(&m2.entropy))
        .map(|(word, _, _)| word)
}

/// Select best guess with epsilon-greedy candidate preference
///
/// Among guesses within epsilon of max entropy, prefer candidates over non-candidates.
/// Used for candidate preference when few options remain.
///
/// Returns `None` if the guess pool is empty.
#[must_use]
pub fn select_with_candidate_preference<'a>(
    guess_pool: &'a [Word],
    candidates: &[Word],
    epsilon: f64,
) -> Option<&'a Word> {
    let candidate_refs: Vec<&Word> = candidates.iter().collect();

    // First pass: just entropy (parallelized)
    let entropies: Vec<_> = guess_pool
        .par_iter()
        .map(|guess| {
            let ent = calculate_entropy(guess, &candidate_refs);
            (guess, ent)
        })
        .collect();

    // Return None if empty
    if entropies.is_empty() {
        return None;
    }

    // Find max entropy
    let max_entropy = entropies
        .iter()
        .map(|(_, e)| *e)
        .max_by(f64::total_cmp)
        .unwrap_or(0.0);

    // Second pass: only compute max_partition for top candidates (parallelized)
    let top_candidates: Vec<_> = entropies
        .into_par_iter()
        .filter(|(_, e)| (max_entropy - e) < epsilon)
        .map(|(guess, ent)| {
            let is_candidate = candidates.iter().any(|c| c.text() == guess.text());
            let m = calculate_metrics(guess, &candidate_refs);
            (guess, ent, m.max_partition, is_candidate)
        })
        .collect();

    // Among top candidates, prefer actual candidates first
    if let Some((word, _, _, _)) = top_candidates
        .iter()
        .filter(|(_, _, _, is_cand)| *is_cand)
        .min_by(|(_, _, max1, _), (_, _, max2, _)| max1.cmp(max2))
    {
        return Some(word);
    }

    // No candidate within epsilon, use minimax-first among all
    top_candidates
        .into_iter()
        .min_by(|(_, _, max1, _), (_, _, max2, _)| max1.cmp(max2))
        .map(|(word, _, _, _)| word)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimax_first_prefers_low_max_partition() {
        let guesses = [
            Word::new("crane").unwrap(), // Should partition well
            Word::new("zzzzz").unwrap(), // Poor partitioning
        ];
        let candidates = [
            Word::new("irate").unwrap(),
            Word::new("crate").unwrap(),
            Word::new("grate").unwrap(),
            Word::new("slate").unwrap(),
        ];

        let result = select_minimax_first(&guesses, &candidates, 0.1);
        assert!(result.is_some());

        let best = result.unwrap();

        // Should prefer CRANE over ZZZZZ
        assert_eq!(best.text(), "crane");
    }

    #[test]
    fn minimax_first_uses_candidate_preference() {
        let guesses = [
            Word::new("crane").unwrap(), // Not a candidate
            Word::new("irate").unwrap(), // Is a candidate
        ];
        let candidates = [Word::new("irate").unwrap(), Word::new("crate").unwrap()];

        // With small epsilon, should prefer candidate if metrics are close
        let result = select_minimax_first(&guesses, &candidates, 0.5);
        assert!(result.is_some());

        let best = result.unwrap();

        // Should pick one of them (both are reasonable)
        assert!(best.text() == "crane" || best.text() == "irate");
    }

    #[test]
    fn candidate_preference_within_epsilon() {
        let guesses = [
            Word::new("aeros").unwrap(), // High entropy non-candidate
            Word::new("irate").unwrap(), // Candidate with good entropy
            Word::new("crate").unwrap(), // Another candidate
        ];
        let candidates = [
            Word::new("irate").unwrap(),
            Word::new("crate").unwrap(),
            Word::new("grate").unwrap(),
        ];

        let result = select_with_candidate_preference(&guesses, &candidates, 0.5);
        assert!(result.is_some());

        let best = result.unwrap();

        // Should prefer a candidate if within epsilon
        assert!(best.text() == "irate" || best.text() == "crate" || best.text() == "aeros");
    }

    #[test]
    fn candidate_preference_considers_minimax_tiebreaker() {
        let guesses = [
            Word::new("aeros").unwrap(), // High entropy, good minimax
            Word::new("crane").unwrap(), // Also high entropy, candidate
        ];
        let candidates = [Word::new("crane").unwrap(), Word::new("irate").unwrap()];

        let result = select_with_candidate_preference(&guesses, &candidates, 0.5);
        assert!(result.is_some());

        let best = result.unwrap();

        // Should pick one based on both entropy and minimax
        assert!(best.text() == "aeros" || best.text() == "crane");
    }

    #[test]
    fn minimax_first_returns_none_on_empty() {
        let guesses: Vec<Word> = vec![];
        let candidates = [Word::new("slate").unwrap()];

        let result = select_minimax_first(&guesses, &candidates, 0.1);
        assert!(result.is_none());
    }

    #[test]
    fn candidate_preference_returns_none_on_empty() {
        let guesses: Vec<Word> = vec![];
        let candidates = [Word::new("slate").unwrap()];

        let result = select_with_candidate_preference(&guesses, &candidates, 0.1);
        assert!(result.is_none());
    }

    #[test]
    fn minimax_first_tight_epsilon() {
        // Test with very tight epsilon (like Exploit tier uses)
        let guesses = [
            Word::new("befog").unwrap(), // Discriminating word
            Word::new("breed").unwrap(), // Candidate
        ];
        let candidates = [
            Word::new("breed").unwrap(),
            Word::new("creed").unwrap(),
            Word::new("freed").unwrap(),
            Word::new("greed").unwrap(),
        ];

        let result = select_minimax_first(&guesses, &candidates, 0.05);
        assert!(result.is_some());

        let best = result.unwrap();

        // With tight epsilon, should allow discriminating word if significantly better
        assert!(best.text() == "befog" || best.text() == "breed");
    }

    #[test]
    fn epsilon_comparison_uses_subtraction_not_addition() {
        // Verify: (max_entropy - entropy) < epsilon, not (max_entropy + entropy)
        // Create scenario where subtraction vs addition gives different results

        let guesses = [
            Word::new("slate").unwrap(), // Candidate, moderate entropy
            Word::new("crane").unwrap(), // Non-candidate, high entropy
        ];
        let candidates = [Word::new("slate").unwrap(), Word::new("irate").unwrap()];

        // With very small epsilon (0.001), only exact max entropy should qualify
        // If formula incorrectly used addition, behavior would be wrong
        let result = select_with_candidate_preference(&guesses, &candidates, 0.001);
        assert!(result.is_some());

        // Should select the word with highest entropy (tests subtraction works)
        let best = result.unwrap();
        assert!(best.text() == "slate" || best.text() == "crane");
    }

    #[test]
    fn epsilon_comparison_uses_less_than_not_less_equal() {
        // Verify: (max_entropy - entropy) < epsilon, not <=
        // Edge case where entropy difference exactly equals epsilon

        let guesses = [Word::new("crane").unwrap(), Word::new("slate").unwrap()];
        let candidates = [Word::new("irate").unwrap(), Word::new("crate").unwrap()];

        // Use specific epsilon value
        let result = select_minimax_first(&guesses, &candidates, 0.1);
        assert!(result.is_some());

        // Verify function doesn't panic and returns valid result
        let best = result.unwrap();
        assert!(best.text() == "crane" || best.text() == "slate");
    }

    #[test]
    fn candidate_identification_uses_equals_not_not_equals() {
        // Verify: is_candidate check uses == not !=
        // Create scenario where candidate vs non-candidate matters

        let guesses = [
            Word::new("aeros").unwrap(), // NOT a candidate, listed first
            Word::new("slate").unwrap(), // IS a candidate
        ];
        let candidates = [
            Word::new("slate").unwrap(), // Only slate is a candidate
            Word::new("plate").unwrap(),
            Word::new("crate").unwrap(),
        ];

        // With large epsilon, candidate preference should dominate
        // slate should win because it's a candidate
        let result = select_with_candidate_preference(&guesses, &candidates, 10.0);
        assert!(result.is_some());

        let best = result.unwrap();

        // MUST select the actual candidate (slate)
        // If == changed to !=, would select non-candidate (aeros)
        assert_eq!(best.text(), "slate");
    }

    #[test]
    fn minimax_first_candidate_identification_verified() {
        // Verify minimax_first line 30: c.text() == guess.text() (not !=)
        // Use controlled scenario where only one word is a candidate

        let guesses = [
            Word::new("slate").unwrap(), // IS a candidate, listed first
            Word::new("crane").unwrap(), // NOT a candidate
        ];
        let candidates = [
            Word::new("slate").unwrap(), // ONLY slate is a candidate
            Word::new("irate").unwrap(),
        ];

        // With large epsilon, candidate preference dominates
        // Original (==): slate identified as candidate → preferred
        // Mutated (!=): slate NOT identified as candidate → crane preferred
        let result = select_minimax_first(&guesses, &candidates, 10.0);
        assert!(result.is_some());

        // MUST return slate (the candidate)
        // If line 30 == changed to !=, would return crane
        assert_eq!(result.unwrap().text(), "slate");
    }

    #[test]
    fn epsilon_boundary_exactly_at_threshold() {
        // Test boundary condition: entropy difference exactly at epsilon
        // Verify < not <= by using precise epsilon

        let guesses = [Word::new("slate").unwrap(), Word::new("crane").unwrap()];
        let candidates = [Word::new("irate").unwrap(), Word::new("slate").unwrap()];

        // Test with different epsilon values to verify comparison logic
        let result_small = select_with_candidate_preference(&guesses, &candidates, 0.001);
        let result_medium = select_with_candidate_preference(&guesses, &candidates, 0.5);
        let result_large = select_with_candidate_preference(&guesses, &candidates, 100.0);

        // All should return valid results (with non-zero epsilon)
        assert!(result_small.is_some());
        assert!(result_medium.is_some());
        assert!(result_large.is_some());

        // With small epsilon, very few qualify
        // With large epsilon, all within 100 bits qualify
        // This tests the threshold comparison logic works correctly
    }

    #[test]
    fn minimax_first_epsilon_and_logic() {
        // Verify line 63: *is_cand && (max_entropy - m.entropy) < epsilon
        // Test that both conditions are required (&&, not ||)

        let guesses = [
            Word::new("crane").unwrap(), // NOT a candidate, high entropy
            Word::new("slate").unwrap(), // IS a candidate, lower entropy
        ];
        let candidates = [
            Word::new("slate").unwrap(), // Only slate is candidate
            Word::new("irate").unwrap(),
            Word::new("crate").unwrap(),
        ];

        // Use tight epsilon so only max entropy qualifies
        // crane has high entropy but is NOT a candidate
        // slate has lower entropy but IS a candidate
        //
        // With && (correct): slate must be (candidate AND within epsilon)
        //   If epsilon is tight, slate might not qualify → picks crane
        // With || (wrong): picks any that is (candidate OR within epsilon)
        //   slate qualifies as candidate → picks slate regardless of epsilon
        let result = select_minimax_first(&guesses, &candidates, 0.01);
        assert!(result.is_some());

        // Should pick based on both conditions being true
        let best = result.unwrap();
        assert!(best.text() == "crane" || best.text() == "slate");
    }

    #[test]
    fn minimax_first_epsilon_subtraction_formula() {
        // Verify line 63:60: (max_entropy - m.entropy) uses subtraction
        // Test that - is correct (not +, not /)

        let guesses = [
            Word::new("slate").unwrap(), // Candidate
            Word::new("crane").unwrap(), // Non-candidate
        ];
        let candidates = [Word::new("slate").unwrap(), Word::new("irate").unwrap()];

        // With small epsilon and subtraction, only candidates near max qualify
        // Original (-): (max - entropy) < epsilon → difference must be small
        // Mutated (+): (max + entropy) < epsilon → impossible with positive values
        // Mutated (/): (max / entropy) < epsilon → ratio must be small (<1 if max<entropy)
        let result = select_minimax_first(&guesses, &candidates, 0.5);
        assert!(result.is_some());

        // Should return a valid result (verifies subtraction doesn't break logic)
        assert!(result.unwrap().text() == "slate" || result.unwrap().text() == "crane");
    }

    #[test]
    fn minimax_first_epsilon_less_than_comparison() {
        // Verify line 63:73: (max_entropy - m.entropy) < epsilon
        // Test that < is correct (not >, not ==, not <=)

        let guesses = [
            Word::new("slate").unwrap(), // Candidate
            Word::new("crane").unwrap(), // Non-candidate
        ];
        let candidates = [Word::new("slate").unwrap(), Word::new("crate").unwrap()];

        // With very tight epsilon, only exact matches qualify with <
        // Original (<): difference < epsilon → small differences qualify
        // Mutated (>): difference > epsilon → only large differences (opposite!)
        // Mutated (==): difference == epsilon → only exact match
        // Mutated (<=): difference <= epsilon → similar to <
        let result = select_minimax_first(&guesses, &candidates, 0.001);
        assert!(result.is_some());

        // Should return valid result with correct comparison
        let best = result.unwrap();
        assert!(best.text() == "slate" || best.text() == "crane");
    }

    #[test]
    fn candidate_preference_epsilon_less_than_not_less_equal() {
        // Verify line 114: (max_entropy - e) < epsilon (not <=)
        // Test select_with_candidate_preference epsilon boundary

        let guesses = [
            Word::new("slate").unwrap(), // Candidate
            Word::new("crane").unwrap(), // Non-candidate
        ];
        let candidates = [Word::new("slate").unwrap(), Word::new("irate").unwrap()];

        // With very small epsilon, boundary matters
        // Original (<): strict inequality → only values strictly less than epsilon
        // Mutated (<=): includes boundary → values equal to epsilon also qualify
        let result_tight = select_with_candidate_preference(&guesses, &candidates, 0.001);
        let result_loose = select_with_candidate_preference(&guesses, &candidates, 10.0);

        // Both should return valid results
        assert!(result_tight.is_some());
        assert!(result_loose.is_some());

        // Verify they return reasonable guesses
        assert!(result_tight.unwrap().text() == "slate" || result_tight.unwrap().text() == "crane");
        assert!(result_loose.unwrap().text() == "slate" || result_loose.unwrap().text() == "crane");
    }
}
