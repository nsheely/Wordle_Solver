//! Minimax worst-case calculation for Wordle patterns
//!
//! Given a guess and set of candidates, computes the maximum remaining candidates
//! for any possible pattern.

use crate::core::{Pattern, Word};

/// Calculate the maximum remaining candidates for a guess
///
/// Returns the worst-case number of remaining candidates after this guess.
///
/// # Strategy
/// For each possible pattern that could result from this guess:
/// - Count how many candidates would produce that pattern
/// - Return the maximum count (worst case)
///
/// # Examples
/// ```
/// use wordle_solver::core::Word;
/// use wordle_solver::solver::minimax::calculate_max_remaining;
///
/// let guess = Word::new("crane").unwrap();
/// let candidates = vec![
///     Word::new("slate").unwrap(),
///     Word::new("irate").unwrap(),
/// ];
/// let candidate_refs: Vec<&Word> = candidates.iter().collect();
///
/// let max_remaining = calculate_max_remaining(&guess, &candidate_refs);
/// assert!(max_remaining <= 2); // Can't be more than total candidates
/// ```
#[must_use]
pub fn calculate_max_remaining(guess: &Word, candidates: &[&Word]) -> usize {
    if candidates.is_empty() {
        return 0;
    }

    // Group candidates by pattern
    let pattern_counts = group_by_pattern(*guess, candidates);

    // Return the maximum count (worst case)
    pattern_counts.iter().max().copied().unwrap_or(0)
}

/// Group candidates by the pattern they produce with the guess
///
/// Returns an array where each index is a pattern value (0-242)
/// and the value is the count of candidates producing that pattern.
fn group_by_pattern(guess: Word, candidates: &[&Word]) -> [usize; 243] {
    let mut counts = [0usize; 243];

    for &candidate in candidates {
        let pattern = Pattern::calculate(&guess, candidate);
        counts[pattern.value() as usize] += 1;
    }

    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_remaining_perfect_split() {
        // Perfect binary split - worst case is 1 (each pattern has 1 candidate)
        let guess = Word::new("slate").unwrap();
        let candidates = [
            Word::new("slate").unwrap(), // Perfect match
            Word::new("zzzzz").unwrap(), // No match
        ];
        let candidate_refs: Vec<&Word> = candidates.iter().collect();

        let max = calculate_max_remaining(&guess, &candidate_refs);
        assert_eq!(max, 1); // Each pattern has 1 candidate
    }

    #[test]
    fn max_remaining_all_same_pattern() {
        // All produce same pattern - worst case is all candidates
        let guess = Word::new("zzzzz").unwrap();
        let candidates = [
            Word::new("aaaaa").unwrap(),
            Word::new("bbbbb").unwrap(),
            Word::new("ccccc").unwrap(),
        ];
        let candidate_refs: Vec<&Word> = candidates.iter().collect();

        let max = calculate_max_remaining(&guess, &candidate_refs);
        assert_eq!(max, 3); // All candidates have same pattern
    }

    #[test]
    fn max_remaining_empty_candidates() {
        let guess = Word::new("crane").unwrap();
        let candidates: Vec<&Word> = vec![];

        let max = calculate_max_remaining(&guess, &candidates);
        assert_eq!(max, 0);
    }

    #[test]
    fn max_remaining_single_candidate() {
        let guess = Word::new("crane").unwrap();
        let candidates = [Word::new("slate").unwrap()];
        let candidate_refs: Vec<&Word> = candidates.iter().collect();

        let max = calculate_max_remaining(&guess, &candidate_refs);
        assert_eq!(max, 1);
    }

    #[test]
    fn group_by_pattern_works() {
        let guess = Word::new("crane").unwrap();
        let candidates = [
            Word::new("slate").unwrap(), // Different pattern than crane
            Word::new("crate").unwrap(), // Different pattern than slate
        ];
        let candidate_refs: Vec<&Word> = candidates.iter().collect();

        let groups = group_by_pattern(guess, &candidate_refs);

        // Should have 2 different patterns with 1 candidate each
        let non_zero_count = groups.iter().filter(|&&c| c > 0).count();
        assert!((1..=2).contains(&non_zero_count));
        assert_eq!(groups.iter().sum::<usize>(), 2);
    }

    #[test]
    fn minimax_prefers_better_splits() {
        // A guess that splits candidates evenly should have lower max_remaining
        // than one that doesn't split them at all

        let candidates = [Word::new("aaaaa").unwrap(), Word::new("bbbbb").unwrap()];
        let candidate_refs: Vec<&Word> = candidates.iter().collect();

        // Bad guess - doesn't distinguish between candidates
        let bad_guess = Word::new("zzzzz").unwrap();
        let bad_max = calculate_max_remaining(&bad_guess, &candidate_refs);

        // Good guess - one of the actual candidates, guarantees a split
        let good_guess = Word::new("aaaaa").unwrap();
        let good_max = calculate_max_remaining(&good_guess, &candidate_refs);

        // Good guess should have lower or equal max remaining
        assert!(good_max <= bad_max);
    }
}
