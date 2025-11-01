//! Wordle Solver - CLI
//!
//! Wordle solver with TUI and CLI modes using information theory and game theory.
//! Performance: 99.64% optimal (3.4333 avg guesses)

use anyhow::Result;
use clap::{Parser, Subcommand};
use wordle_solver::{
    commands::{
        SolveConfig, analyze_word, print_test_all_statistics, run_benchmark, run_simple,
        run_test_all, solve_word,
    },
    core::Word,
    output::{print_analysis_result, print_benchmark_result, print_solve_result},
    solver::{Solver, Strategy, StrategyType},
    wordlists::{ALLOWED, ANSWERS, loader::words_from_slice},
};

#[derive(Parser)]
#[command(
    name = "wordle_solver",
    about = "Wordle solver using adaptive information-theoretic strategies (99.64% optimal)",
    version,
    author
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Strategy: adaptive (default), entropy, minimax, hybrid, random
    #[arg(short, long, global = true, default_value = "adaptive")]
    strategy: String,

    /// Wordlist: 'all' (default, 12972 words), 'answers' (2315 only), or path to file
    #[arg(short = 'w', long, global = true, default_value = "all")]
    wordlist: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Interactive TUI mode (default - beautiful visualizations)
    Play,

    /// Simple CLI mode (interactive solver without TUI)
    Simple,

    /// Solve a specific target word
    Solve {
        /// The target word to solve
        word: String,

        /// Show verbose output with candidate counts
        #[arg(short, long)]
        verbose: bool,
    },

    /// Analyze the entropy of a specific word
    Analyze {
        /// Word to analyze
        word: String,
    },

    /// Benchmark solver performance
    Benchmark {
        /// Number of random words to test
        #[arg(short = 'n', long, default_value = "50")]
        count: usize,

        /// Override first word (default: SALET in full mode, auto in answers-only)
        #[arg(short = 'f', long)]
        first_word: Option<String>,
    },

    /// Test solver on ALL possible answers
    TestAll {
        /// Limit number of words to test
        #[arg(short, long)]
        limit: Option<usize>,

        /// Override first word (default: SALET in full mode, auto in answers-only)
        #[arg(short = 'f', long)]
        first_word: Option<String>,
    },
}

/// Load wordlists based on the -w flag
///
/// Returns (`guess_pool`, `answer_candidates`)
/// - "all": Use all 12,972 words for guessing, 2,315 as candidates
/// - "answers": Use only 2,315 words for both (demonstrates exploration paradox)
/// - "<path>": Load custom wordlist from file
fn load_wordlists(wordlist_mode: &str) -> Result<(Vec<Word>, Vec<Word>)> {
    use wordle_solver::wordlists::loader::load_from_file;

    match wordlist_mode {
        "all" => {
            // Default: full search space
            let all_words = words_from_slice(ALLOWED);
            let answer_words = words_from_slice(ANSWERS);
            Ok((all_words, answer_words))
        }
        "answers" => {
            // Answers-only mode: demonstrates exploration paradox
            let answer_words = words_from_slice(ANSWERS);
            Ok((answer_words.clone(), answer_words))
        }
        path => {
            // Load from custom file - use for both guessing and answers
            let custom_words = load_from_file(path)?;
            Ok((custom_words.clone(), custom_words))
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Load word lists based on -w flag
    let (all_words, answer_words) = load_wordlists(&cli.wordlist)?;

    // Default to Play mode if no command given
    let command = cli.command.unwrap_or(Commands::Play);

    match command {
        Commands::Play => run_play_command(&all_words, &answer_words),
        Commands::Simple => run_simple_command(&cli.strategy, &all_words, &answer_words),
        Commands::Solve { word, verbose } => {
            run_solve_command(&cli.strategy, &word, verbose, &all_words, &answer_words)
        }
        Commands::Analyze { word } => run_analyze_command(&word, &all_words, &answer_words),
        Commands::Benchmark { count, first_word } => {
            run_benchmark_command(
                &cli.strategy,
                count,
                first_word.as_deref(),
                &all_words,
                &answer_words,
            );
            Ok(())
        }
        Commands::TestAll { limit, first_word } => {
            run_test_all_command(
                &cli.strategy,
                limit,
                first_word.as_deref(),
                &all_words,
                &answer_words,
            );
            Ok(())
        }
    }
}

fn run_solve_command(
    strategy_name: &str,
    word: &str,
    verbose: bool,
    all_words: &[Word],
    answer_words: &[Word],
) -> Result<()> {
    let strategy = StrategyType::from_name(strategy_name);
    let solver = Solver::new(strategy, all_words, answer_words);
    solve_command(word, verbose, &solver)
}

fn solve_command<S: Strategy>(word: &str, verbose: bool, solver: &Solver<S>) -> Result<()> {
    let config = SolveConfig::new(word.to_string());
    let result = solve_word(config, solver).map_err(|e| anyhow::anyhow!(e))?;

    print_solve_result(&result, verbose);
    Ok(())
}

fn run_analyze_command(word: &str, all_words: &[Word], answer_words: &[Word]) -> Result<()> {
    let result = analyze_word(word, all_words, answer_words).map_err(|e| anyhow::anyhow!(e))?;
    print_analysis_result(&result);
    Ok(())
}

fn run_benchmark_command(
    strategy_name: &str,
    count: usize,
    first_word: Option<&str>,
    all_words: &[Word],
    answer_words: &[Word],
) {
    let strategy = StrategyType::from_name(strategy_name);
    let solver = Solver::new(strategy, all_words, answer_words);
    benchmark_command(count, first_word, &solver, all_words, answer_words);
}

fn benchmark_command<S: Strategy>(
    count: usize,
    first_word: Option<&str>,
    solver: &Solver<S>,
    all_words: &[Word],
    answer_words: &[Word],
) {
    if let Some(word_str) = first_word {
        println!("Running benchmark on {count} random words with forced first word: {word_str}...");
    } else {
        println!("Running benchmark on {count} random words...");
    }

    // Take first N words from answer list
    let test_words: Vec<Word> = answer_words.iter().take(count).cloned().collect();

    // Convert first_word to Word if provided
    let forced_first =
        first_word.and_then(|word_str| all_words.iter().find(|w| w.text() == word_str));

    let result = run_benchmark(solver, &test_words, forced_first);
    print_benchmark_result(&result);
}

fn run_test_all_command(
    strategy_name: &str,
    limit: Option<usize>,
    first_word: Option<&str>,
    all_words: &[Word],
    answer_words: &[Word],
) {
    println!("\n{}", "═".repeat(70));
    println!(" Comprehensive Wordle Solver Test ");
    println!("{}", "═".repeat(70));
    println!("\nTesting against {} possible answers", answer_words.len());
    println!("Strategy: {strategy_name}");
    if let Some(word) = first_word {
        println!("Forced first word: {word}");
    }
    println!();

    // Convert first_word to Word if provided
    let forced_first =
        first_word.and_then(|word_str| all_words.iter().find(|w| w.text() == word_str));

    let strategy = StrategyType::from_name(strategy_name);
    let solver = Solver::new(strategy, all_words, answer_words);
    let stats = run_test_all(&solver, answer_words, limit, forced_first);
    print_test_all_statistics(&stats);
}

fn run_simple_command(
    strategy_name: &str,
    all_words: &[Word],
    answer_words: &[Word],
) -> Result<()> {
    let strategy = StrategyType::from_name(strategy_name);
    let solver = Solver::new(strategy, all_words, answer_words);
    run_simple(&solver).map_err(|e| anyhow::anyhow!(e))
}

fn run_play_command(all_words: &[Word], answer_words: &[Word]) -> Result<()> {
    use wordle_solver::interactive::{App, run_tui};

    let app = App::new(all_words, answer_words);
    run_tui(app)
}
