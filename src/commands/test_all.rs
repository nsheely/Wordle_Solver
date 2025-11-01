//! Test all answers - comprehensive solver evaluation
//!
//! Runs the solver against every possible answer word and generates statistics.

use crate::core::{Pattern, Word};
use crate::solver::{Solver, Strategy};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Result from testing a single word
#[derive(Debug, Clone)]
pub struct WordTestResult {
    pub word: String,
    pub guesses: Vec<String>,
    pub num_guesses: usize,
    pub success: bool,
    pub duration: Duration,
}

/// Statistics from testing all words
#[derive(Debug)]
pub struct TestAllStatistics {
    pub total_words: usize,
    pub solved: usize,
    pub failed: usize,
    pub guess_distribution: HashMap<usize, usize>,
    pub total_time: Duration,
    pub average_guesses: f64,
    pub max_guesses: usize,
    pub min_guesses: usize,
    pub best_word: Option<(String, usize)>,
    pub worst_words: Vec<(String, usize)>,
    pub first_guess_used: HashMap<String, usize>,
}

/// Run solver on all answer words (or a limited subset)
///
/// If `forced_first` is provided, it will be used as the first guess instead of
/// letting the solver choose.
///
/// # Panics
///
/// May panic if the solver encounters an impossible state (e.g., no valid guesses remaining).
#[allow(clippy::too_many_lines)] // Complex test orchestration
pub fn run_test_all<S: Strategy>(
    wordle_solver: &Solver<S>,
    answer_words: &[Word],
    limit: Option<usize>,
    forced_first: Option<&Word>,
) -> TestAllStatistics {
    let test_words: Vec<&Word> = answer_words
        .iter()
        .take(limit.unwrap_or(answer_words.len()))
        .collect();

    println!("🎯 Testing {} words...", test_words.len());

    // Progress bar
    let pb = ProgressBar::new(test_words.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) | {msg}")
            .unwrap()
            .progress_chars("█▓▒░"),
    );

    let mut results = Vec::new();
    let mut guess_distribution: HashMap<usize, usize> = HashMap::new();
    let mut first_guess_used: HashMap<String, usize> = HashMap::new();

    let total_start = Instant::now();

    for (idx, &answer_word) in test_words.iter().enumerate() {
        let word_start = Instant::now();
        let mut history: Vec<(Word, Pattern)> = Vec::new();
        let mut guesses = Vec::new();
        let mut success = false;

        for turn in 1..=6 {
            // Get next guess
            let guess = if let (1, Some(forced)) = (turn, forced_first) {
                // Use forced first word on first turn
                forced
            } else {
                // Otherwise use solver
                match wordle_solver.next_guess(&history) {
                    Some(g) => g,
                    None => break, // No candidates remaining
                }
            };

            let guess_text = guess.text().to_string();
            guesses.push(guess_text.clone());

            // Track first guess
            if guesses.len() == 1 {
                *first_guess_used.entry(guess_text.clone()).or_insert(0) += 1;
            }

            // Check if correct
            if guess.text() == answer_word.text() {
                success = true;
                break;
            }

            // Calculate pattern
            let pattern = Pattern::calculate(guess, answer_word);

            // Add to history
            history.push((guess.clone(), pattern));
        }

        let num_guesses = guesses.len();
        let duration = word_start.elapsed();

        results.push(WordTestResult {
            word: answer_word.text().to_string(),
            guesses,
            num_guesses,
            success,
            duration,
        });

        if success {
            *guess_distribution.entry(num_guesses).or_insert(0) += 1;
        }

        // Update progress
        if idx % 10 == 0 && !results.is_empty() {
            let avg =
                results.iter().map(|r| r.num_guesses).sum::<usize>() as f64 / results.len() as f64;
            pb.set_message(format!("Avg: {avg:.2}"));
        }
        pb.inc(1);
    }

    pb.finish_with_message("Complete!");

    let total_time = total_start.elapsed();

    // Calculate statistics
    let solved_count = results.iter().filter(|r| r.success).count();
    let failed_count = results.len() - solved_count;

    let total_guesses: usize = results
        .iter()
        .filter(|r| r.success)
        .map(|r| r.num_guesses)
        .sum();
    let average_guesses = if solved_count > 0 {
        total_guesses as f64 / solved_count as f64
    } else {
        0.0
    };

    let max_guesses = results
        .iter()
        .filter(|r| r.success)
        .map(|r| r.num_guesses)
        .max()
        .unwrap_or(0);

    let min_guesses = results
        .iter()
        .filter(|r| r.success)
        .map(|r| r.num_guesses)
        .min()
        .unwrap_or(0);

    let best_word = results
        .iter()
        .filter(|r| r.success)
        .min_by_key(|r| r.num_guesses)
        .map(|r| (r.word.clone(), r.num_guesses));

    let mut worst_words: Vec<(String, usize)> = results
        .iter()
        .filter(|r| r.success)
        .filter(|r| r.num_guesses >= 5)
        .map(|r| (r.word.clone(), r.num_guesses))
        .collect();
    worst_words.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    worst_words.truncate(10);

    TestAllStatistics {
        total_words: results.len(),
        solved: solved_count,
        failed: failed_count,
        guess_distribution,
        total_time,
        average_guesses,
        max_guesses,
        min_guesses,
        best_word,
        worst_words,
        first_guess_used,
    }
}

/// Print test-all statistics with beautiful formatting
#[allow(clippy::too_many_lines)] // Comprehensive output formatting
pub fn print_test_all_statistics(stats: &TestAllStatistics) {
    println!("\n{}", "═".repeat(70));
    println!(" Test Results ");
    println!("{}", "═".repeat(70));

    // Overall performance
    println!("\n📊 {}", "Overall Performance".bright_cyan().bold());
    println!("  Total words tested:  {}", stats.total_words);
    println!(
        "  Successfully solved: {} {}",
        stats.solved,
        format!(
            "({:.1}%)",
            stats.solved as f64 / stats.total_words as f64 * 100.0
        )
        .green()
    );
    if stats.failed > 0 {
        println!(
            "  Failed to solve:     {} {}",
            stats.failed,
            format!(
                "({:.1}%)",
                stats.failed as f64 / stats.total_words as f64 * 100.0
            )
            .red()
        );
    }
    println!(
        "  Average guesses:     {}",
        format!("{:.3}", stats.average_guesses)
            .bright_yellow()
            .bold()
    );
    println!(
        "  Total time:          {:.2}s",
        stats.total_time.as_secs_f64()
    );
    println!(
        "  Time per word:       {:.1}ms",
        stats.total_time.as_millis() as f64 / stats.total_words as f64
    );

    // Guess distribution
    println!("\n📈 {}", "Guess Distribution".bright_cyan().bold());
    let max_count = *stats.guess_distribution.values().max().unwrap_or(&1);
    for guesses in 1..=6 {
        let count = stats.guess_distribution.get(&guesses).unwrap_or(&0);
        if stats.solved > 0 {
            let percentage = *count as f64 / stats.solved as f64 * 100.0;
            let bar_len = if max_count > 0 {
                (*count * 40 / max_count).max(usize::from(*count > 0))
            } else {
                0
            };
            let bar = format!(
                "{}{}",
                "█".repeat(bar_len).green(),
                "░".repeat(40_usize.saturating_sub(bar_len)).bright_black()
            );

            println!("  {guesses} guesses: {bar} {count:4} ({percentage:5.1}%)");
        }
    }

    // Information theory metrics
    println!("\n🧮 Information Theory Metrics");
    let total_bits = (stats.total_words as f64).log2();
    let bits_per_guess = if stats.average_guesses > 0.0 {
        total_bits / stats.average_guesses
    } else {
        0.0
    };
    let theoretical_max_bits = 5.835; // SALET's entropy
    let efficiency = if theoretical_max_bits > 0.0 {
        (bits_per_guess / theoretical_max_bits) * 100.0
    } else {
        0.0
    };
    println!("  Total information:   {total_bits:.2} bits");
    println!("  Bits per guess:      {bits_per_guess:.2} bits");
    println!(
        "  Efficiency:          {efficiency:.1}% (vs theoretical max {theoretical_max_bits:.2} bits/guess)"
    );

    // Best and worst cases
    if let Some((word, guesses)) = &stats.best_word {
        println!("\n✨ {}", "Best Performance".green().bold());
        println!(
            "  {} solved in {} guess{}",
            word.to_uppercase().bright_green(),
            guesses,
            if *guesses == 1 { "" } else { "es" }
        );
    }

    if !stats.worst_words.is_empty() {
        println!("\n😰 {}", "Hardest Words (5-6 guesses)".yellow().bold());
        for (word, guesses) in stats.worst_words.iter().take(5) {
            println!("  {} ({} guesses)", word.to_uppercase().yellow(), guesses);
        }
    }

    // First guess analysis
    println!("\n🎯 First Guess Usage");
    let mut first_guesses: Vec<(String, usize)> = stats
        .first_guess_used
        .iter()
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    first_guesses.sort_by_key(|(_, count)| std::cmp::Reverse(*count));

    for (word, count) in first_guesses.iter().take(5) {
        let percentage = *count as f64 / stats.total_words as f64 * 100.0;
        println!(
            "  {}: {} times ({:.1}%)",
            word.to_uppercase(),
            count,
            percentage
        );
    }

    // Theoretical comparison
    println!("\n📐 {}", "Theoretical Comparison".bright_cyan().bold());
    println!(
        "  Our average:         {} guesses",
        format!("{:.3}", stats.average_guesses)
            .bright_yellow()
            .bold()
    );
    println!("  Theoretical optimal: 3.421 guesses (Bertsimas & Paskov 2022, 10,657-word pool)");

    let difference = stats.average_guesses - 3.421;
    let diff_str = format!("{difference:+.3} guesses");
    let colored_diff = if difference.abs() < 0.02 {
        diff_str.green()
    } else if difference.abs() < 0.05 {
        diff_str.yellow()
    } else {
        diff_str.red()
    };
    println!("  Difference:          {colored_diff}");

    let performance = 3.421 / stats.average_guesses * 100.0;
    let perf_str = format!("{performance:.1}% of optimal");
    let colored_perf = if performance >= 99.7 {
        perf_str.bright_green().bold()
    } else if performance >= 99.0 {
        perf_str.yellow()
    } else {
        perf_str.red()
    };
    println!("  Performance:         {colored_perf}");
}
