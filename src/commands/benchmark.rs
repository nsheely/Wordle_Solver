//! Benchmark command
//!
//! Tests solver performance across multiple words.

use crate::core::{Pattern, Word};
use crate::solver::{Solver, Strategy};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Result of a benchmark run
pub struct BenchmarkResult {
    pub total_words: usize,
    pub total_guesses: usize,
    pub average_guesses: f64,
    pub min_guesses: usize,
    pub max_guesses: usize,
    pub distribution: HashMap<usize, usize>,
    pub duration: Duration,
    pub words_per_second: f64,
}

/// Run benchmark on a set of target words
///
/// If `forced_first` is provided, it will be used as the first guess instead of
/// letting the solver choose.
pub fn run_benchmark<S: Strategy>(
    solver: &Solver<S>,
    target_words: &[Word],
    forced_first: Option<&Word>,
) -> BenchmarkResult {
    let start = Instant::now();
    let mut total_guesses = 0;
    let mut min_guesses = usize::MAX;
    let mut max_guesses = 0;
    let mut distribution: HashMap<usize, usize> = HashMap::new();

    for target in target_words {
        let mut history: Vec<(Word, Pattern)> = Vec::new();
        let mut guesses = 0;

        loop {
            guesses += 1;

            let guess = if let (1, Some(forced)) = (guesses, forced_first) {
                // Use forced first word on first guess
                forced
            } else {
                // Otherwise use solver
                match solver.next_guess(&history) {
                    Some(g) => g,
                    None => break,
                }
            };

            let pattern = Pattern::calculate(guess, target);
            history.push((*guess, pattern));

            if pattern.is_perfect() || guesses >= 6 {
                break;
            }
        }

        total_guesses += guesses;
        min_guesses = min_guesses.min(guesses);
        max_guesses = max_guesses.max(guesses);
        *distribution.entry(guesses).or_insert(0) += 1;
    }

    let duration = start.elapsed();
    let total_words = target_words.len();

    BenchmarkResult {
        total_words,
        total_guesses,
        average_guesses: total_guesses as f64 / total_words as f64,
        min_guesses,
        max_guesses,
        distribution,
        duration,
        words_per_second: total_words as f64 / duration.as_secs_f64(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::EntropyStrategy;
    use crate::wordlists::loader::words_from_slice;
    use crate::wordlists::{ALLOWED, ANSWERS};

    #[test]
    fn benchmark_runs() {
        let all_words = words_from_slice(&ALLOWED[..100]);
        let answer_words = words_from_slice(&ANSWERS[..10]);

        let solver = Solver::new(EntropyStrategy, &all_words, &answer_words);
        let result = run_benchmark(&solver, &answer_words, None);

        assert_eq!(result.total_words, 10);
        assert!(result.total_guesses > 0);
        assert!(result.average_guesses >= 1.0);
        assert!(result.min_guesses >= 1);
        assert!(result.max_guesses <= 6);
    }

    #[test]
    fn benchmark_distribution_sums_correctly() {
        let all_words = words_from_slice(&ALLOWED[..100]);
        let answer_words = words_from_slice(&ANSWERS[..10]);

        let solver = Solver::new(EntropyStrategy, &all_words, &answer_words);
        let result = run_benchmark(&solver, &answer_words, None);

        let distribution_sum: usize = result.distribution.values().sum();
        assert_eq!(distribution_sum, result.total_words);
    }

    #[test]
    fn benchmark_with_forced_first_word() {
        let all_words = words_from_slice(&ALLOWED[..100]);
        let answer_words = words_from_slice(&ANSWERS[..5]);

        let solver = Solver::new(EntropyStrategy, &all_words, &answer_words);
        let forced = all_words.first();

        let result = run_benchmark(&solver, &answer_words, forced);

        assert_eq!(result.total_words, 5);
        assert!(result.average_guesses >= 1.0);
    }

    #[test]
    fn benchmark_empty_word_list() {
        let all_words = words_from_slice(&ALLOWED[..100]);
        let answer_words: Vec<Word> = vec![];

        let solver = Solver::new(EntropyStrategy, &all_words, &answer_words);
        let result = run_benchmark(&solver, &answer_words, None);

        assert_eq!(result.total_words, 0);
        assert_eq!(result.total_guesses, 0);
    }

    #[test]
    fn benchmark_metrics_consistency() {
        let all_words = words_from_slice(&ALLOWED[..100]);
        let answer_words = words_from_slice(&ANSWERS[..10]);

        let solver = Solver::new(EntropyStrategy, &all_words, &answer_words);
        let result = run_benchmark(&solver, &answer_words, None);

        // Average should be between min and max
        assert!(result.average_guesses >= result.min_guesses as f64);
        assert!(result.average_guesses <= result.max_guesses as f64);

        // Distribution should only contain valid guess counts (1-6)
        for &guess_count in result.distribution.keys() {
            assert!((1..=6).contains(&guess_count));
        }
    }
}
