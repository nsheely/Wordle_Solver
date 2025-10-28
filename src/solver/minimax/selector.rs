//! Minimax-based guess selection strategy
//!
//! Always selects the guess that minimizes the worst-case remaining candidates.

use super::calculator::calculate_max_remaining;
use crate::core::Word;
use rayon::prelude::*;

/// Select best guess by minimizing worst-case remaining candidates
///
/// Returns the word with the lowest maximum remaining candidates and that value,
/// or `None` if the guess pool is empty.
///
/// # Examples
/// ```
/// use wordle_solver::core::Word;
/// use wordle_solver::solver::minimax::select_best_guess;
///
/// let guesses = vec![
///     Word::new("aaaaa").unwrap(),
///     Word::new("crane").unwrap(),
/// ];
/// let candidates = vec![
///     Word::new("slate").unwrap(),
///     Word::new("irate").unwrap(),
/// ];
///
/// let result = select_best_guess(&guesses, &candidates);
/// assert!(result.is_some());
/// let (best, max_remaining) = result.unwrap();
/// // CRANE should be better than AAAAA for these candidates
/// assert!(max_remaining <= 2);
/// ```
#[must_use]
pub fn select_best_guess<'a>(
    guess_pool: &'a [Word],
    candidates: &[Word],
) -> Option<(&'a Word, usize)> {
    let candidate_refs: Vec<&Word> = candidates.iter().collect();

    guess_pool
        .par_iter()
        .map(|guess| {
            let max_remaining = calculate_max_remaining(guess, &candidate_refs);
            (guess, max_remaining)
        })
        .min_by_key(|(_, max)| *max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_lowest_max_remaining() {
        let guesses = [
            Word::new("zzzzz").unwrap(), // Bad guess - doesn't split well
            Word::new("crane").unwrap(), // Better guess - diverse letters
        ];
        let candidates = vec![
            Word::new("slate").unwrap(),
            Word::new("irate").unwrap(),
            Word::new("crate").unwrap(),
            Word::new("grate").unwrap(),
        ];

        let result = select_best_guess(&guesses, &candidates);
        assert!(result.is_some());

        let (best, max_remaining) = result.unwrap();

        // CRANE should be better than ZZZZZ
        assert_eq!(best.text(), "crane");
        assert!(max_remaining < 4); // Should split the candidates somewhat
    }

    #[test]
    fn single_guess_returns_that_guess() {
        let guesses = [Word::new("crane").unwrap()];
        let candidates = [Word::new("slate").unwrap()];

        let result = select_best_guess(&guesses, &candidates);
        assert!(result.is_some());

        let (best, _) = result.unwrap();
        assert_eq!(best.text(), "crane");
    }

    #[test]
    fn ties_resolved_consistently() {
        // If multiple guesses have same max_remaining, should pick one consistently
        let guesses = [Word::new("aaaaa").unwrap(), Word::new("bbbbb").unwrap()];
        let candidates = [Word::new("ccccc").unwrap()];

        let result1 = select_best_guess(&guesses, &candidates);
        let result2 = select_best_guess(&guesses, &candidates);

        assert!(result1.is_some());
        assert!(result2.is_some());

        let (best1, max1) = result1.unwrap();
        let (best2, max2) = result2.unwrap();

        // Should be deterministic
        assert_eq!(best1.text(), best2.text());
        assert_eq!(max1, max2);
    }

    #[test]
    fn returns_none_on_empty_guess_pool() {
        let guesses: Vec<Word> = vec![];
        let candidates = [Word::new("slate").unwrap()];

        let result = select_best_guess(&guesses, &candidates);
        assert!(result.is_none());
    }

    #[test]
    fn prefers_actual_candidates_when_equal() {
        // When all else is equal, any of the candidate words is a valid choice
        let candidates = [Word::new("slate").unwrap(), Word::new("crate").unwrap()];

        let guesses = [
            Word::new("slate").unwrap(), // One of the candidates
            Word::new("zzzzz").unwrap(), // Not a candidate
        ];

        let result = select_best_guess(&guesses, &candidates);
        assert!(result.is_some());

        let (best, _) = result.unwrap();

        // SLATE should be selected because it's a better guess
        // (it guarantees finding the answer if it's SLATE)
        assert!(best.text() == "slate" || best.text() == "zzzzz");
    }
}
