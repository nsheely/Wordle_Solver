//! Pure entropy-based word selection
//!
//! Selects words that maximize Shannon entropy (expected information gain).

use super::calculator::calculate_entropy;
use crate::core::Word;
use rayon::prelude::*;

/// Select best guess by maximizing entropy
///
/// Returns the word with highest entropy and its entropy value,
/// or `None` if the guess pool is empty.
///
/// # Examples
/// ```
/// use wordle_solver::core::Word;
/// use wordle_solver::solver::entropy::select_best_guess;
///
/// let guesses = vec![
///     Word::new("aaaaa").unwrap(),
///     Word::new("aeros").unwrap(),
/// ];
/// let candidates = vec![
///     Word::new("slate").unwrap(),
///     Word::new("irate").unwrap(),
/// ];
///
/// let result = select_best_guess(&guesses, &candidates);
/// assert!(result.is_some());
/// let (best, entropy) = result.unwrap();
/// assert_eq!(best.text(), "aeros"); // AEROS has higher entropy than AAAAA
/// assert!(entropy > 0.0);
/// ```
#[must_use]
pub fn select_best_guess<'a>(
    guess_pool: &'a [Word],
    candidates: &[Word],
) -> Option<(&'a Word, f64)> {
    let candidate_refs: Vec<&Word> = candidates.iter().collect();

    guess_pool
        .par_iter()
        .map(|guess| {
            let entropy = calculate_entropy(guess, &candidate_refs);
            (guess, entropy)
        })
        .max_by(|(_, e1), (_, e2)| e1.total_cmp(e2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_highest_entropy() {
        let guesses = [
            Word::new("aaaaa").unwrap(), // Low entropy (all same letter)
            Word::new("aeros").unwrap(), // Higher entropy (diverse letters)
        ];
        let candidates = vec![
            Word::new("slate").unwrap(),
            Word::new("irate").unwrap(),
            Word::new("crate").unwrap(),
            Word::new("grate").unwrap(),
        ];

        let result = select_best_guess(&guesses, &candidates);
        assert!(result.is_some());

        let (best, entropy) = result.unwrap();

        // AEROS should have higher entropy than AAAAA
        assert_eq!(best.text(), "aeros");
        assert!(entropy > 0.5); // Should be reasonably high
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
        // If multiple guesses have same entropy, should pick one consistently
        let guesses = [Word::new("aaaaa").unwrap(), Word::new("bbbbb").unwrap()];
        let candidates = [Word::new("ccccc").unwrap()];

        let result1 = select_best_guess(&guesses, &candidates);
        let result2 = select_best_guess(&guesses, &candidates);

        assert!(result1.is_some());
        assert!(result2.is_some());

        let (best1, entropy1) = result1.unwrap();
        let (best2, entropy2) = result2.unwrap();

        // Should be deterministic
        assert_eq!(best1.text(), best2.text());
        assert!((entropy1 - entropy2).abs() < 0.001);
    }

    #[test]
    fn returns_none_on_empty_guess_pool() {
        let guesses: Vec<Word> = vec![];
        let candidates = [Word::new("slate").unwrap()];

        let result = select_best_guess(&guesses, &candidates);
        assert!(result.is_none());
    }
}
