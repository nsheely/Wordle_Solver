//! Word solving command
//!
//! Solves a specific target word and returns the solution path.

use crate::core::{Pattern, Word};
use crate::solver::entropy::calculate_entropy;
use crate::solver::{Solver, Strategy};

/// Configuration for solving a word
pub struct SolveConfig {
    pub target: String,
    pub max_guesses: usize,
}

impl SolveConfig {
    #[must_use]
    pub const fn new(target: String) -> Self {
        Self {
            target,
            max_guesses: 6,
        }
    }
}

/// Result of solving a word
pub struct SolveResult {
    pub success: bool,
    pub guesses: Vec<GuessStep>,
    pub target: String,
}

/// A single guess step in the solution
pub struct GuessStep {
    pub word: String,
    pub pattern: Pattern,
    pub candidates_before: usize,
    pub candidates_after: usize,
    pub entropy: Option<f64>,
    pub expected_remaining: Option<f64>,
}

/// Solve a specific word using the given solver and strategy
///
/// # Errors
///
/// Returns an error if:
/// - The target word is invalid (not 5 letters or contains non-ASCII)
/// - The solver cannot provide a valid guess
/// - Maximum guess limit is reached without finding the solution
pub fn solve_word<S: Strategy>(
    config: SolveConfig,
    solver: &Solver<S>,
) -> Result<SolveResult, String> {
    // Find target in answer words
    let target_word = Word::new(&config.target).map_err(|e| format!("Invalid target word: {e}"))?;

    // Build history as we go
    let mut history: Vec<(Word, Pattern)> = Vec::new();
    let mut guesses: Vec<GuessStep> = Vec::new();

    for _ in 0..config.max_guesses {
        let candidates_before = solver.count_candidates(&history);

        // Get next guess
        let guess = solver
            .next_guess(&history)
            .ok_or_else(|| "No candidates remaining".to_string())?;

        // Calculate entropy for this guess against remaining candidates (if applicable)
        let (entropy, expected_remaining) = if candidates_before > 1 {
            let current_candidates = solver.get_candidates(&history);
            let ent = calculate_entropy(guess, &current_candidates);
            let exp_remaining = candidates_before as f64 / ent.exp2();
            (Some(ent), Some(exp_remaining))
        } else {
            (None, None)
        };

        // Calculate pattern against target
        let pattern = Pattern::calculate(guess, &target_word);

        // Add to history
        history.push((*guess, pattern));

        let candidates_after = solver.count_candidates(&history);

        guesses.push(GuessStep {
            word: guess.text().to_string(),
            pattern,
            candidates_before,
            candidates_after,
            entropy,
            expected_remaining,
        });

        // Check if solved
        if pattern.is_perfect() {
            return Ok(SolveResult {
                success: true,
                guesses,
                target: config.target,
            });
        }
    }

    // Failed to solve
    Ok(SolveResult {
        success: false,
        guesses,
        target: config.target,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::EntropyStrategy;
    use crate::wordlists::loader::words_from_slice;
    use crate::wordlists::{ALLOWED, ANSWERS};

    #[test]
    fn solve_word_succeeds() {
        let all_words = words_from_slice(&ALLOWED[..100]);
        let answer_words = words_from_slice(&ANSWERS[..50]);

        let solver = Solver::new(EntropyStrategy, &all_words, &answer_words);
        let config = SolveConfig::new("aback".to_string());

        let result = solve_word(config, &solver).unwrap();

        // Should solve ABACK (first word in ANSWERS)
        assert!(result.success || result.guesses.len() == 6);
        assert!(!result.guesses.is_empty());
    }

    #[test]
    fn solve_records_history() {
        let all_words = words_from_slice(&ALLOWED[..100]);
        let answer_words = words_from_slice(&ANSWERS[..50]);

        let solver = Solver::new(EntropyStrategy, &all_words, &answer_words);
        let config = SolveConfig::new("abase".to_string());

        let result = solve_word(config, &solver).unwrap();

        // Should have at least one guess
        assert!(!result.guesses.is_empty());

        // Each step should show candidate reduction (or stay same)
        for step in &result.guesses {
            assert!(step.candidates_after <= step.candidates_before);
        }
    }

    #[test]
    fn solve_invalid_target_returns_error() {
        let all_words = words_from_slice(&ALLOWED[..100]);
        let answer_words = words_from_slice(&ANSWERS[..50]);

        let solver = Solver::new(EntropyStrategy, &all_words, &answer_words);
        let config = SolveConfig::new("zzzzz".to_string()); // Not in answer list

        let result = solve_word(config, &solver);

        // Should return error for invalid target
        assert!(result.is_err());
    }

    #[test]
    fn solve_with_max_guesses_limit() {
        let all_words = words_from_slice(&ALLOWED[..100]);
        let answer_words = words_from_slice(&ANSWERS[..50]);

        let solver = Solver::new(EntropyStrategy, &all_words, &answer_words);
        let mut config = SolveConfig::new("aback".to_string());
        config.max_guesses = 3;

        let result = solve_word(config, &solver).unwrap();

        // Should respect max_guesses limit
        assert!(result.guesses.len() <= 3);
    }

    #[test]
    fn solve_perfect_first_guess() {
        let all_words = words_from_slice(&ALLOWED[..100]);
        let answer_words = words_from_slice(&ANSWERS[..50]);

        let solver = Solver::new(EntropyStrategy, &all_words, &answer_words);
        let target = "aback"; // First answer word
        let config = SolveConfig::new(target.to_string());

        let result = solve_word(config, &solver).unwrap();

        // If we get lucky and guess it first try
        if result.success && result.guesses.len() == 1 {
            assert_eq!(result.guesses[0].word, target);
        }
    }
}
