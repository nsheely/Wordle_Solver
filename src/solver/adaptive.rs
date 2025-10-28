//! Adaptive strategy
//!
//! Adjusts tactics based on number of remaining candidates.

use super::{selection, strategy::Strategy};
use crate::core::Word;

/// Adaptive strategy with configurable tier thresholds
///
/// Achieves 99.7-99.8% optimal performance (3.428-3.436 avg guesses) by using different
/// tactics depending on how many candidates remain.
///
/// ## How Thresholds Work
///
/// Thresholds use cascading `>` comparisons:
/// ```text
/// if candidates > pure_entropy_threshold          → PureEntropy
/// else if candidates > entropy_minimax_threshold  → EntropyMinimax
/// else if candidates > hybrid_threshold           → Hybrid
/// else if candidates > minimax_first_threshold    → MinimaxFirst
/// else                                            → Random
/// ```
///
/// With default thresholds (100, 21, 9, 2):
/// - **101+ candidates**: `PureEntropy` - Pure entropy maximization
/// - **22-100 candidates**: `EntropyMinimax` - Entropy + minimax tiebreakers
/// - **10-21 candidates**: `Hybrid` - Hybrid scoring (entropy × 100) - (`max_partition` × 10)
/// - **3-9 candidates**: `MinimaxFirst` - Minimax-first with 0.1 epsilon
/// - **1-2 candidates**: `Random` - Random selection from candidates
#[derive(Debug, Clone)]
pub struct AdaptiveStrategy {
    /// Candidates > this use `PureEntropy` (default: 100)
    pub pure_entropy_threshold: usize,

    /// Candidates > this use `EntropyMinimax` (default: 21)
    pub entropy_minimax_threshold: usize,

    /// Candidates > this use `Hybrid` (default: 9)
    pub hybrid_threshold: usize,

    /// Candidates > this use `MinimaxFirst` (default: 2)
    pub minimax_first_threshold: usize,
}

impl AdaptiveStrategy {
    /// Create a new adaptive strategy with custom thresholds
    #[must_use]
    pub const fn new(
        pure_entropy_threshold: usize,
        entropy_minimax_threshold: usize,
        hybrid_threshold: usize,
        minimax_first_threshold: usize,
    ) -> Self {
        Self {
            pure_entropy_threshold,
            entropy_minimax_threshold,
            hybrid_threshold,
            minimax_first_threshold,
        }
    }

    /// Get the current tier based on number of candidates
    #[must_use]
    pub const fn get_tier(&self, num_candidates: usize) -> AdaptiveTier {
        if num_candidates > self.pure_entropy_threshold {
            AdaptiveTier::PureEntropy
        } else if num_candidates > self.entropy_minimax_threshold {
            AdaptiveTier::EntropyMinimax
        } else if num_candidates > self.hybrid_threshold {
            AdaptiveTier::Hybrid
        } else if num_candidates > self.minimax_first_threshold {
            AdaptiveTier::MinimaxFirst
        } else {
            AdaptiveTier::Random
        }
    }
}

impl Default for AdaptiveStrategy {
    /// Default thresholds tuned for 99.7-99.8% optimal performance
    fn default() -> Self {
        Self::new(
            100, // pure_entropy_threshold: 101+ candidates
            21,  // entropy_minimax_threshold: 22-100 candidates
            9,   // hybrid_threshold: 10-21 candidates
            2,   // minimax_first_threshold: 3-9 candidates (1-2 use Random)
        )
    }
}

/// The current tier/phase of the adaptive strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveTier {
    /// Many candidates (101+): Pure entropy maximization
    PureEntropy,

    /// Medium candidates (22-100): Entropy + minimax tiebreakers
    EntropyMinimax,

    /// Few candidates (10-21): Hybrid scoring
    Hybrid,

    /// Very few (3-9): Minimax-first with candidate preference
    MinimaxFirst,

    /// Endgame (1-2): Random selection from candidates
    Random,
}

impl Strategy for AdaptiveStrategy {
    fn select_guess<'a>(&self, guess_pool: &'a [Word], candidates: &[Word]) -> Option<&'a Word> {
        let tier = self.get_tier(candidates.len());

        match tier {
            AdaptiveTier::PureEntropy => {
                // 101+ candidates: Pure entropy maximization
                super::entropy::select_best_guess(guess_pool, candidates).map(|(best, _)| best)
            }

            AdaptiveTier::EntropyMinimax => {
                // 22-100 candidates: Entropy + minimax tiebreakers
                selection::select_with_expected_tiebreaker(guess_pool, candidates)
            }

            AdaptiveTier::Hybrid => {
                // 10-21 candidates: Hybrid scoring
                selection::select_with_hybrid_scoring(guess_pool, candidates)
            }

            AdaptiveTier::MinimaxFirst => {
                // 3-9 candidates: Minimax-first with 0.1 epsilon
                selection::select_minimax_first(guess_pool, candidates, 0.1)
            }

            AdaptiveTier::Random => {
                // 1-2 candidates: Random selection
                super::strategy::RandomStrategy.select_guess(guess_pool, candidates)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_tiers_correct() {
        let strategy = AdaptiveStrategy::default();

        assert_eq!(strategy.get_tier(200), AdaptiveTier::PureEntropy);
        assert_eq!(strategy.get_tier(101), AdaptiveTier::PureEntropy);
        assert_eq!(strategy.get_tier(100), AdaptiveTier::EntropyMinimax);
        assert_eq!(strategy.get_tier(50), AdaptiveTier::EntropyMinimax);
        assert_eq!(strategy.get_tier(22), AdaptiveTier::EntropyMinimax);
        assert_eq!(strategy.get_tier(21), AdaptiveTier::Hybrid);
        assert_eq!(strategy.get_tier(15), AdaptiveTier::Hybrid);
        assert_eq!(strategy.get_tier(10), AdaptiveTier::Hybrid);
        assert_eq!(strategy.get_tier(9), AdaptiveTier::MinimaxFirst);
        assert_eq!(strategy.get_tier(5), AdaptiveTier::MinimaxFirst);
        assert_eq!(strategy.get_tier(4), AdaptiveTier::MinimaxFirst);
        assert_eq!(strategy.get_tier(3), AdaptiveTier::MinimaxFirst);
        assert_eq!(strategy.get_tier(2), AdaptiveTier::Random);
        assert_eq!(strategy.get_tier(1), AdaptiveTier::Random);
    }

    #[test]
    fn adaptive_custom_thresholds() {
        let strategy = AdaptiveStrategy::new(50, 20, 10, 5);

        assert_eq!(strategy.get_tier(100), AdaptiveTier::PureEntropy);
        assert_eq!(strategy.get_tier(51), AdaptiveTier::PureEntropy);
        assert_eq!(strategy.get_tier(50), AdaptiveTier::EntropyMinimax);
        assert_eq!(strategy.get_tier(21), AdaptiveTier::EntropyMinimax);
        assert_eq!(strategy.get_tier(20), AdaptiveTier::Hybrid);
        assert_eq!(strategy.get_tier(11), AdaptiveTier::Hybrid);
        assert_eq!(strategy.get_tier(10), AdaptiveTier::MinimaxFirst);
        assert_eq!(strategy.get_tier(6), AdaptiveTier::MinimaxFirst);
        assert_eq!(strategy.get_tier(5), AdaptiveTier::Random);
    }

    #[test]
    fn adaptive_selects_candidate_when_few_remain() {
        let guess_pool = vec![
            Word::new("crane").unwrap(),
            Word::new("slate").unwrap(),
            Word::new("irate").unwrap(),
        ];

        let candidates = vec![Word::new("irate").unwrap()];

        let strategy = AdaptiveStrategy::default();
        let result = strategy.select_guess(&guess_pool, &candidates);

        assert!(result.is_some());
        let guess = result.unwrap();

        // With 1 candidate, should select it
        assert_eq!(guess.text(), "irate");
    }
}
