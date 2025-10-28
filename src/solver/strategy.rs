//! Guess selection strategies
//!
//! Defines the Strategy trait and concrete implementations.

use super::AdaptiveStrategy;
use crate::core::Word;

/// A strategy for selecting the best guess from a pool of candidates
pub trait Strategy {
    /// Select the best guess from the guess pool given the current candidates
    ///
    /// Returns the best guess, or `None` if the guess pool is empty.
    fn select_guess<'a>(&self, guess_pool: &'a [Word], candidates: &[Word]) -> Option<&'a Word>;
}

/// Enum wrapper for all strategy types
///
/// Allows runtime selection of strategy while maintaining static dispatch.
pub enum StrategyType {
    /// Adaptive strategy (default, best performance)
    Adaptive(AdaptiveStrategy),
    /// Pure entropy maximization
    Entropy(EntropyStrategy),
    /// Pure minimax optimization
    Minimax(MinimaxStrategy),
    /// Hybrid entropy/minimax
    Hybrid(HybridStrategy),
    /// Random selection from candidates
    Random(RandomStrategy),
}

impl Strategy for StrategyType {
    fn select_guess<'a>(&self, guess_pool: &'a [Word], candidates: &[Word]) -> Option<&'a Word> {
        match self {
            Self::Adaptive(s) => s.select_guess(guess_pool, candidates),
            Self::Entropy(s) => s.select_guess(guess_pool, candidates),
            Self::Minimax(s) => s.select_guess(guess_pool, candidates),
            Self::Hybrid(s) => s.select_guess(guess_pool, candidates),
            Self::Random(s) => s.select_guess(guess_pool, candidates),
        }
    }
}

impl StrategyType {
    /// Create strategy from name string
    ///
    /// Supported names: "adaptive", "entropy", "pure-entropy", "minimax", "hybrid", "random"
    /// Defaults to adaptive if name is unrecognized.
    #[must_use]
    pub fn from_name(name: &str) -> Self {
        match name {
            "entropy" | "pure-entropy" => Self::Entropy(EntropyStrategy),
            "minimax" => Self::Minimax(MinimaxStrategy),
            "hybrid" => Self::Hybrid(HybridStrategy::default()),
            "random" => Self::Random(RandomStrategy),
            _ => Self::Adaptive(AdaptiveStrategy::default()),
        }
    }
}

/// Pure entropy maximization strategy
///
/// Always selects the guess with the highest Shannon entropy.
pub struct EntropyStrategy;

impl Strategy for EntropyStrategy {
    fn select_guess<'a>(&self, guess_pool: &'a [Word], candidates: &[Word]) -> Option<&'a Word> {
        super::entropy::select_best_guess(guess_pool, candidates).map(|(best, _)| best)
    }
}

/// Pure minimax strategy
///
/// Always selects the guess that minimizes worst-case remaining candidates.
pub struct MinimaxStrategy;

impl Strategy for MinimaxStrategy {
    fn select_guess<'a>(&self, guess_pool: &'a [Word], candidates: &[Word]) -> Option<&'a Word> {
        super::minimax::select_best_guess(guess_pool, candidates).map(|(best, _)| best)
    }
}

/// Hybrid strategy combining entropy and minimax
///
/// Uses entropy when many candidates remain, switches to minimax near the end.
pub struct HybridStrategy {
    /// Switch to minimax when candidates <= this threshold
    pub minimax_threshold: usize,
}

impl HybridStrategy {
    /// Create a new hybrid strategy
    ///
    /// # Parameters
    /// - `minimax_threshold`: Switch to minimax when candidates <= this value (default: 5)
    #[must_use]
    pub const fn new(minimax_threshold: usize) -> Self {
        Self { minimax_threshold }
    }
}

impl Default for HybridStrategy {
    fn default() -> Self {
        Self::new(5)
    }
}

impl Strategy for HybridStrategy {
    fn select_guess<'a>(&self, guess_pool: &'a [Word], candidates: &[Word]) -> Option<&'a Word> {
        if candidates.len() <= self.minimax_threshold {
            super::minimax::select_best_guess(guess_pool, candidates).map(|(best, _)| best)
        } else {
            super::entropy::select_best_guess(guess_pool, candidates).map(|(best, _)| best)
        }
    }
}

/// Random strategy
///
/// Randomly selects from remaining candidates. Useful for endgame when only 1-2 candidates remain.
pub struct RandomStrategy;

impl Strategy for RandomStrategy {
    fn select_guess<'a>(&self, guess_pool: &'a [Word], candidates: &[Word]) -> Option<&'a Word> {
        use rand::prelude::IndexedRandom;

        // Prefer candidates from the guess pool
        let valid_candidates: Vec<&Word> = candidates
            .iter()
            .filter(|c| guess_pool.iter().any(|g| g.text() == c.text()))
            .collect();

        if let Some(candidate) = valid_candidates.choose(&mut rand::rng()) {
            guess_pool.iter().find(|w| w.text() == candidate.text())
        } else {
            // Fallback: pick first candidate if none are in guess pool
            candidates
                .first()
                .and_then(|c| guess_pool.iter().find(|w| w.text() == c.text()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_test_data() -> (Vec<Word>, Vec<Word>) {
        let guesses = vec![Word::new("crane").unwrap(), Word::new("slate").unwrap()];
        let candidates = vec![
            Word::new("irate").unwrap(),
            Word::new("crate").unwrap(),
            Word::new("grate").unwrap(),
        ];
        (guesses, candidates)
    }

    #[test]
    fn entropy_strategy_selects_guess() {
        let (guesses, candidates) = setup_test_data();

        let strategy = EntropyStrategy;
        let result = strategy.select_guess(&guesses, &candidates);

        assert!(result.is_some());
        let guess = result.unwrap();

        // Should return one of the guesses
        assert!(guess.text() == "crane" || guess.text() == "slate");
    }

    #[test]
    fn minimax_strategy_selects_guess() {
        let (guesses, candidates) = setup_test_data();

        let strategy = MinimaxStrategy;
        let result = strategy.select_guess(&guesses, &candidates);

        assert!(result.is_some());
        let guess = result.unwrap();

        // Should return one of the guesses
        assert!(guess.text() == "crane" || guess.text() == "slate");
    }

    #[test]
    fn hybrid_uses_entropy_for_many_candidates() {
        let (guesses, candidates) = setup_test_data();

        // 3 candidates, threshold = 2, should use entropy
        let strategy = HybridStrategy::new(2);
        let result = strategy.select_guess(&guesses, &candidates);

        assert!(result.is_some());
        let guess = result.unwrap();

        // Should return one of the guesses (using entropy)
        assert!(guess.text() == "crane" || guess.text() == "slate");
    }

    #[test]
    fn hybrid_uses_minimax_for_few_candidates() {
        let (guesses, candidates) = setup_test_data();

        // 3 candidates, threshold = 5, should use minimax
        let strategy = HybridStrategy::new(5);
        let result = strategy.select_guess(&guesses, &candidates);

        assert!(result.is_some());
        let guess = result.unwrap();

        // Should return one of the guesses (using minimax)
        assert!(guess.text() == "crane" || guess.text() == "slate");
    }

    #[test]
    fn hybrid_default_threshold() {
        let strategy = HybridStrategy::default();
        assert_eq!(strategy.minimax_threshold, 5);
    }

    #[test]
    fn random_strategy_selects_from_candidates() {
        let guesses = vec![
            Word::new("crane").unwrap(),
            Word::new("slate").unwrap(),
            Word::new("irate").unwrap(),
        ];
        let candidates = vec![Word::new("irate").unwrap()];

        let strategy = RandomStrategy;
        let result = strategy.select_guess(&guesses, &candidates);

        assert!(result.is_some());
        let guess = result.unwrap();

        // Should select the only candidate
        assert_eq!(guess.text(), "irate");
    }
}
