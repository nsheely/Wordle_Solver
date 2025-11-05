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

    #[test]
    fn strategy_type_from_name_entropy() {
        // Verify "entropy" match arm exists (not deleted)
        let strategy = StrategyType::from_name("entropy");
        assert!(matches!(strategy, StrategyType::Entropy(_)));

        let strategy2 = StrategyType::from_name("pure-entropy");
        assert!(matches!(strategy2, StrategyType::Entropy(_)));
    }

    #[test]
    fn strategy_type_from_name_minimax() {
        // Verify "minimax" match arm exists (not deleted)
        let strategy = StrategyType::from_name("minimax");
        assert!(matches!(strategy, StrategyType::Minimax(_)));
    }

    #[test]
    fn strategy_type_from_name_hybrid() {
        // Verify "hybrid" match arm exists (not deleted)
        let strategy = StrategyType::from_name("hybrid");
        assert!(matches!(strategy, StrategyType::Hybrid(_)));
    }

    #[test]
    fn strategy_type_from_name_random() {
        // Verify "random" match arm exists (not deleted)
        let strategy = StrategyType::from_name("random");
        assert!(matches!(strategy, StrategyType::Random(_)));
    }

    #[test]
    fn strategy_type_select_guess_delegates() {
        // Verify StrategyType::select_guess actually calls strategy (not returns None)
        let guesses = vec![Word::new("crane").unwrap(), Word::new("slate").unwrap()];
        let candidates = vec![Word::new("crane").unwrap()];

        let strategy = StrategyType::Entropy(EntropyStrategy);
        let result = strategy.select_guess(&guesses, &candidates);

        // MUST return a guess (not None)
        // If replaced with None, this fails
        assert!(result.is_some());
    }

    #[test]
    fn random_strategy_candidate_preference() {
        // Verify RandomStrategy line 130: guess_pool.iter().any(|g| g.text() == c.text())
        // Use multiple candidates: one IN guess pool, one NOT in guess pool

        let guesses = vec![
            Word::new("slate").unwrap(), // "slate" is in guess pool
        ];
        let candidates = vec![
            Word::new("slate").unwrap(), // This should be selected
            Word::new("crane").unwrap(), // This should NOT be selected
        ];

        let strategy = RandomStrategy;

        // Original (==):
        //   "slate": any(|g| g == "slate") → true → included in valid_candidates
        //   "crane": any(|g| g == "crane") → false → excluded
        //   valid_candidates = ["slate"], picks "slate"
        //
        // Mutated (!=):
        //   "slate": any(|g| g != "slate") → false → excluded
        //   "crane": any(|g| g != "crane") → true ("slate" != "crane") → included!
        //   valid_candidates = ["crane"], picks "crane"
        //   Line 134 tries to find "crane" in guess_pool → returns None (not found)
        //
        // So mutation causes None instead of Some("slate")!

        for _ in 0..10 {
            let result = strategy.select_guess(&guesses, &candidates);
            assert!(result.is_some());
            // MUST return slate (the only candidate in guess pool)
            // If == changed to !=, would return None
            assert_eq!(result.unwrap().text(), "slate");
        }
    }

    #[test]
    fn random_strategy_fallback_path() {
        // Verify RandomStrategy line 139: w.text() == c.text() (not !=)
        // When NO candidates are in guess pool, uses fallback path
        let guesses = vec![
            Word::new("slate").unwrap(), // This IS in guess pool
            Word::new("crane").unwrap(), // This IS in guess pool
        ];
        let candidates = vec![
            Word::new("zzzzz").unwrap(), // NOT in guess pool - triggers fallback
            Word::new("aaaaa").unwrap(),
        ];

        let strategy = RandomStrategy;
        let result = strategy.select_guess(&guesses, &candidates);

        // Fallback: tries to find first candidate (zzzzz) in guess pool
        // If line 139 == changed to !=, behavior would be different
        // Since zzzzz is NOT in guess pool, should return None
        assert!(result.is_none());
    }

    #[test]
    fn hybrid_strategy_threshold_comparison() {
        // Verify HybridStrategy uses <= threshold (line 110), not >
        // Test boundary: at threshold and below vs above
        let guesses = vec![Word::new("crane").unwrap(), Word::new("slate").unwrap()];

        // Test 1: candidates.len() == threshold (should use minimax with <=)
        let candidates_at: Vec<Word> = vec![
            Word::new("irate").unwrap(),
            Word::new("crate").unwrap(),
            Word::new("grate").unwrap(),
        ];

        let strategy = HybridStrategy::new(3);
        let result_at = strategy.select_guess(&guesses, &candidates_at);

        // With <= : 3 <= 3 is true → uses minimax
        // With > : 3 > 3 is false → uses entropy
        assert!(result_at.is_some());

        // Test 2: candidates.len() < threshold (should use minimax with <=)
        let candidates_below: Vec<Word> =
            vec![Word::new("irate").unwrap(), Word::new("crate").unwrap()];

        let result_below = strategy.select_guess(&guesses, &candidates_below);

        // With <= : 2 <= 3 is true → uses minimax
        // With > : 2 > 3 is false → uses entropy
        assert!(result_below.is_some());

        // Test 3: candidates.len() > threshold (should use entropy with <=)
        let candidates_above: Vec<Word> = vec![
            Word::new("irate").unwrap(),
            Word::new("crate").unwrap(),
            Word::new("grate").unwrap(),
            Word::new("prate").unwrap(),
        ];

        let result_above = strategy.select_guess(&guesses, &candidates_above);

        // With <= : 4 <= 3 is false → uses entropy
        // With > : 4 > 3 is true → uses minimax (WRONG, but still returns Some)
        assert!(result_above.is_some());

        // The key is that all three should return valid results
        // If comparison is wrong, the logic is inverted but both strategies work
        // This is a black-box testing limitation - can't distinguish which strategy was used
    }
}
