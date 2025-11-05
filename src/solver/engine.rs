//! Main Wordle solver interface

use super::strategy::Strategy;
use crate::core::{Pattern, Word};

/// Main Wordle solver
///
/// Coordinates the solving process using a given strategy.
pub struct Solver<'a, S: Strategy> {
    strategy: S,
    all_words: &'a [Word],
    answer_words: &'a [Word],
}

impl<'a, S: Strategy> Solver<'a, S> {
    /// Create a new solver with the given strategy and word lists
    ///
    /// # Parameters
    /// - `strategy`: The guess selection strategy to use
    /// - `all_words`: All valid guessable words
    /// - `answer_words`: Subset of words that can be answers
    pub const fn new(strategy: S, all_words: &'a [Word], answer_words: &'a [Word]) -> Self {
        Self {
            strategy,
            all_words,
            answer_words,
        }
    }

    /// Get the best first guess for a new game
    ///
    /// Returns SALET if available (Bertsimas & Paskov 2022), otherwise uses strategy.
    /// SALET achieves 3.421 average guesses (optimal via dynamic programming, 10,657-word pool).
    ///
    /// Note: SALET has 5.835 bits entropy, which is not the maximum, but it's
    /// optimal for minimizing expected guesses across all possible answers.
    pub fn first_guess(&self) -> Option<&'a Word> {
        // Try to use SALET as the hardcoded optimal first guess
        self.all_words
            .iter()
            .find(|w| w.text() == "salet")
            .or_else(|| {
                // SALET not available (e.g., answers-only mode), use strategy
                self.strategy
                    .select_guess(self.all_words, self.answer_words)
            })
    }

    /// Get the next best guess given previous guesses and patterns
    ///
    /// # Parameters
    /// - `history`: Slice of (guess, pattern) pairs from previous turns
    ///
    /// Returns the best next guess, or None if no candidates remain.
    pub fn next_guess(&self, history: &[(Word, Pattern)]) -> Option<&'a Word> {
        // If this is the first guess, use the hardcoded optimal
        if history.is_empty() {
            return self.first_guess();
        }

        let candidates = self.filter_candidates(history);

        if candidates.is_empty() {
            return None;
        }

        // If only one candidate remains, just guess it
        if candidates.len() == 1 {
            return Some(candidates[0]);
        }

        // Convert candidates to owned Vec<Word> to avoid lifetime issues
        let candidate_words: Vec<Word> = candidates.into_iter().cloned().collect();

        self.strategy.select_guess(self.all_words, &candidate_words)
    }

    /// Filter answer words to those consistent with the guess history
    ///
    /// Returns candidates that would produce the observed patterns for all guesses.
    fn filter_candidates(&self, history: &[(Word, Pattern)]) -> Vec<&'a Word> {
        self.answer_words
            .iter()
            .filter(|&candidate| {
                history.iter().all(|(guess, observed_pattern)| {
                    let pattern = Pattern::calculate(guess, candidate);
                    pattern == *observed_pattern
                })
            })
            .collect()
    }

    /// Count how many candidates remain given the history
    pub fn count_candidates(&self, history: &[(Word, Pattern)]) -> usize {
        self.filter_candidates(history).len()
    }

    /// Get the current candidates (public accessor)
    pub fn get_candidates(&self, history: &[(Word, Pattern)]) -> Vec<&'a Word> {
        self.filter_candidates(history)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::strategy::EntropyStrategy;

    fn setup_solver() -> (Vec<Word>, Vec<Word>) {
        let all_words = vec![
            Word::new("crane").unwrap(),
            Word::new("slate").unwrap(),
            Word::new("irate").unwrap(),
            Word::new("crate").unwrap(),
            Word::new("grate").unwrap(),
        ];
        let answer_words = vec![
            Word::new("irate").unwrap(),
            Word::new("crate").unwrap(),
            Word::new("grate").unwrap(),
        ];
        (all_words, answer_words)
    }

    #[test]
    fn first_guess_returns_valid_word() {
        let (all_words, answer_words) = setup_solver();
        let solver = Solver::new(EntropyStrategy, &all_words, &answer_words);

        let result = solver.first_guess();
        assert!(result.is_some());

        let guess = result.unwrap();
        assert!(all_words.iter().any(|w| w == guess));
    }

    #[test]
    fn next_guess_with_empty_history() {
        let (all_words, answer_words) = setup_solver();
        let solver = Solver::new(EntropyStrategy, &all_words, &answer_words);

        let guess = solver.next_guess(&[]);
        assert!(guess.is_some());
    }

    #[test]
    fn next_guess_filters_candidates() {
        let (all_words, answer_words) = setup_solver();
        let solver = Solver::new(EntropyStrategy, &all_words, &answer_words);

        // Simulate guessing CRANE and getting a specific pattern
        let guess = Word::new("crane").unwrap();
        let answer = Word::new("irate").unwrap();
        let pattern = Pattern::calculate(&guess, &answer);

        let history = vec![(guess, pattern)];
        let next = solver.next_guess(&history);

        assert!(next.is_some());
    }

    #[test]
    fn next_guess_returns_none_when_no_candidates() {
        let (all_words, answer_words) = setup_solver();
        let solver = Solver::new(EntropyStrategy, &all_words, &answer_words);

        // Create an impossible pattern that no word satisfies
        let guess = Word::new("zzzzz").unwrap();
        let pattern = Pattern::PERFECT; // Claim we got all greens for ZZZZZ

        let history = vec![(guess, pattern)];
        let next = solver.next_guess(&history);

        // Should return None because no candidate can match this impossible pattern
        assert!(next.is_none());
    }

    #[test]
    fn count_candidates_decreases() {
        let (all_words, answer_words) = setup_solver();
        let solver = Solver::new(EntropyStrategy, &all_words, &answer_words);

        // Start with all candidates
        assert_eq!(solver.count_candidates(&[]), answer_words.len());

        // Make a guess
        let guess = Word::new("crane").unwrap();
        let answer = Word::new("irate").unwrap();
        let pattern = Pattern::calculate(&guess, &answer);

        let history = vec![(guess, pattern)];
        let remaining = solver.count_candidates(&history);

        // Should have fewer candidates after filtering
        assert!(remaining <= answer_words.len());
    }

    #[test]
    fn filter_candidates_exact_match() {
        let (all_words, answer_words) = setup_solver();
        let solver = Solver::new(EntropyStrategy, &all_words, &answer_words);

        // Guess the exact answer
        let guess = Word::new("irate").unwrap();
        let pattern = Pattern::PERFECT;

        let history = vec![(guess.clone(), pattern)];
        let candidates = solver.filter_candidates(&history);

        // Should have exactly one candidate: IRATE
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].text(), "irate");
    }

    #[test]
    fn filter_candidates_multiple_guesses() {
        let (all_words, answer_words) = setup_solver();
        let solver = Solver::new(EntropyStrategy, &all_words, &answer_words);

        let answer = Word::new("grate").unwrap();

        // First guess: CRANE
        let guess1 = Word::new("crane").unwrap();
        let pattern1 = Pattern::calculate(&guess1, &answer);

        // Second guess: IRATE
        let guess2 = Word::new("irate").unwrap();
        let pattern2 = Pattern::calculate(&guess2, &answer);

        let history = vec![(guess1, pattern1), (guess2, pattern2)];
        let candidates = solver.filter_candidates(&history);

        // Should narrow down significantly
        assert!(candidates.len() <= answer_words.len());
        // GRATE should be in the candidates
        assert!(candidates.iter().any(|&w| w.text() == "grate"));
    }

    #[test]
    fn first_guess_returns_salet_when_available() {
        // Verify first_guess uses == "salet", not != "salet"
        // Create word list with "salet" included
        let all_words = vec![
            Word::new("salet").unwrap(), // Optimal first word
            Word::new("crane").unwrap(),
            Word::new("slate").unwrap(),
        ];
        let answer_words = vec![Word::new("crane").unwrap(), Word::new("slate").unwrap()];
        let solver = Solver::new(EntropyStrategy, &all_words, &answer_words);

        let first = solver.first_guess();
        assert!(first.is_some());

        // MUST return "salet" specifically (optimal hardcoded first guess)
        // If == changed to !=, would return wrong word (crane or slate)
        assert_eq!(first.unwrap().text(), "salet");
    }

    #[test]
    fn next_guess_single_candidate_optimization() {
        // Verify: when exactly 1 candidate, returns it directly (not via strategy)
        // If == changed to !=, behavior differs

        // Setup with multiple possible words
        let all_words = vec![
            Word::new("crane").unwrap(),
            Word::new("slate").unwrap(),
            Word::new("irate").unwrap(),
            Word::new("crate").unwrap(),
        ];
        let answer_words = vec![Word::new("irate").unwrap(), Word::new("crate").unwrap()];
        let solver = Solver::new(EntropyStrategy, &all_words, &answer_words);

        // Narrow down to exactly one candidate with specific guess
        // IRATE vs CRATE gives specific pattern
        let guess1 = Word::new("slate").unwrap();
        let answer = Word::new("crate").unwrap();
        let pattern1 = Pattern::calculate(&guess1, &answer);

        let guess2 = Word::new("crane").unwrap();
        let pattern2 = Pattern::calculate(&guess2, &answer);

        let history = vec![(guess1, pattern1), (guess2, pattern2)];
        let candidates = solver.filter_candidates(&history);

        // If we got exactly one candidate, verify next_guess returns it
        if candidates.len() == 1 && candidates[0].text() == "crate" {
            let next = solver.next_guess(&history);
            assert!(next.is_some());
            // MUST return the single candidate
            // If == changed to !=, might not work correctly
            assert_eq!(next.unwrap().text(), "crate");
        }
    }

    #[test]
    fn get_candidates_returns_filtered_list_not_empty() {
        // Verify get_candidates returns actual filtered candidates, not vec![]
        let (all_words, answer_words) = setup_solver();
        let solver = Solver::new(EntropyStrategy, &all_words, &answer_words);

        // No history = all answer words are candidates
        let candidates = solver.get_candidates(&[]);

        // MUST return non-empty list (all answer words)
        // If replaced with vec![], this would fail
        assert!(!candidates.is_empty());
        assert_eq!(candidates.len(), answer_words.len());
    }

    #[test]
    fn get_candidates_filters_correctly() {
        // Additional verification that get_candidates actually filters
        let (all_words, answer_words) = setup_solver();
        let solver = Solver::new(EntropyStrategy, &all_words, &answer_words);

        let guess = Word::new("crane").unwrap();
        let answer = Word::new("irate").unwrap();
        let pattern = Pattern::calculate(&guess, &answer);

        let history = vec![(guess, pattern)];
        let candidates = solver.get_candidates(&history);

        // Should have fewer candidates after filtering
        assert!(candidates.len() < answer_words.len());
        assert!(!candidates.is_empty());

        // irate should still be a candidate
        assert!(candidates.iter().any(|&w| w.text() == "irate"));
    }
}
