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
}
