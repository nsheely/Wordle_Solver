//! Simple interactive CLI mode
//!
//! Text-based interactive solver without TUI

use crate::core::{Pattern, Word};
use crate::solver::entropy::calculate_metrics;
use crate::solver::{Solver, Strategy};
use std::io::{self, Write};

/// Run the simple interactive CLI mode
///
/// # Errors
///
/// Returns an error if there's an I/O error reading user input or if the solver
/// cannot provide a valid guess.
#[allow(clippy::too_many_lines)] // Interactive game loop requires detailed handling
pub fn run_simple<S: Strategy>(solver: &Solver<S>) -> Result<(), String> {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║              Wordle Solver - Interactive Mode                ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("I'll suggest optimal guesses using information theory and game theory.");
    println!("After each guess, enter the feedback pattern:\n");
    println!("  - Use G/g/🟩 for green (correct position)");
    println!("  - Use Y/y/🟨 for yellow (wrong position)");
    println!("  - Use -/_/⬜ for gray (not in word)");
    println!("  - Or type 'win' if you got it right!\n");
    println!("Commands: 'quit' to exit, 'new' for new game, 'undo' to undo last guess\n");

    let mut history: Vec<(Word, Pattern)> = Vec::new();
    let mut turn = 1;

    loop {
        // Get current candidates count
        let candidates_count = solver.count_candidates(&history);

        if candidates_count == 0 {
            println!("\n❌ No candidates remain! Your feedback may be incorrect.");
            println!("Type 'undo' to go back, or 'new' to start over.\n");

            match get_user_input("Command")? {
                cmd if cmd == "undo" => {
                    if history.pop().is_some() {
                        turn -= 1;
                        println!("✓ Undone! Back to turn {turn}\n");
                    } else {
                        println!("Nothing to undo!\n");
                    }
                }
                cmd if cmd == "new" => {
                    history.clear();
                    turn = 1;
                    println!("\n🔄 New game started!\n");
                    continue;
                }
                _ => continue,
            }
        }

        // Get next guess suggestion
        let guess = solver
            .next_guess(&history)
            .ok_or("No valid guesses available")?;

        println!("────────────────────────────────────────────────────────────");
        println!("Turn {turn}: {candidates_count} candidates remaining");
        println!("────────────────────────────────────────────────────────────");

        // Calculate and display metrics
        let candidates = solver.get_candidates(&history);
        let metrics = calculate_metrics(guess, &candidates);

        println!("\n📊 Suggested guess: {}", guess.text().to_uppercase());
        println!("   Entropy:          {:.3} bits", metrics.entropy);
        println!(
            "   Expected info:    {:.1}x reduction",
            metrics.entropy.exp2()
        );
        println!(
            "   Expected remain:  {:.1} candidates",
            metrics.expected_remaining
        );
        println!(
            "   Worst case:       {} candidates\n",
            metrics.max_partition
        );

        // Show some candidates if count is small
        if candidates_count <= 10 {
            println!("Remaining candidates:");
            for candidate in candidates.iter().take(10) {
                println!("  • {}", candidate.text().to_uppercase());
            }
            println!();
        }

        // Get feedback
        let feedback = loop {
            let input = get_user_input("Enter feedback (G/Y/-, 'win', or command)")?.to_lowercase();

            match input.as_str() {
                "quit" | "q" | "exit" => {
                    println!("\n👋 Thanks for playing!\n");
                    return Ok(());
                }
                "new" | "n" => {
                    history.clear();
                    turn = 0; // Will be incremented to 1
                    println!("\n🔄 New game started!\n");
                    break None;
                }
                "undo" | "u" => {
                    if history.pop().is_some() {
                        turn -= 1;
                        println!("✓ Undone! Back to turn {turn}\n");
                        break None;
                    }
                    println!("Nothing to undo!\n");
                }
                "win" | "correct" | "yes" | "solved" => {
                    // Shortcut for all greens (perfect match)
                    break Some(Pattern::PERFECT);
                }
                _ => {
                    if let Some(pattern) = Pattern::from_str(&input) {
                        break Some(pattern);
                    }
                    println!("❌ Invalid pattern! Use G/Y/-, 'win', or '🟩🟨⬜🟩🟨'\n");
                }
            }
        };

        if let Some(pattern) = feedback {
            // Add to history
            history.push((*guess, pattern));

            // Check if solved
            if pattern.is_perfect() {
                use colored::Colorize;

                // Celebration banner
                println!("\n{}", "═".repeat(70).bright_cyan());
                println!(
                    "{}",
                    "    🎉 🎊 ✨  W O R D L E   S O L V E D !  ✨ 🎊 🎉    "
                        .bright_green()
                        .bold()
                );
                println!("{}", "═".repeat(70).bright_cyan());

                // Victory stats
                let performance = match turn {
                    1 => ("🏆 Perfect!", "Incredible hole-in-one!"),
                    2 => ("⭐ Excellent!", "Outstanding performance!"),
                    3 => ("💫 Great!", "Very well played!"),
                    4 => ("✨ Good!", "Nice work!"),
                    5 => ("👍 Solved!", "Got it!"),
                    _ => ("✓ Complete!", "Success!"),
                };

                println!("\n  {}", performance.0.bright_yellow().bold());
                println!("  {}", performance.1.bright_white());
                println!(
                    "\n  Solution found in {} {}",
                    turn.to_string().bright_cyan().bold(),
                    if turn == 1 { "guess" } else { "guesses" }
                );

                // Show guess history with emojis
                println!("\n  Guess history:");
                for (i, (word, pat)) in history.iter().enumerate() {
                    use crate::output::formatters::pattern_to_emoji;
                    println!(
                        "    {}. {} {}",
                        (i + 1).to_string().bright_black(),
                        word.text().to_uppercase().bright_white().bold(),
                        pattern_to_emoji(*pat)
                    );
                }

                println!("\n{}", "═".repeat(70).bright_cyan());
                println!();

                match get_user_input("Play again? (yes/no)")?
                    .to_lowercase()
                    .as_str()
                {
                    "yes" | "y" => {
                        history.clear();
                        turn = 0;
                        println!("\n🔄 New game started!\n");
                    }
                    _ => {
                        println!("\n👋 Thanks for playing!\n");
                        return Ok(());
                    }
                }
            }

            turn += 1;
        }
    }
}

/// Get user input with a prompt
fn get_user_input(prompt: &str) -> Result<String, String> {
    print!("{prompt}: ");
    io::stdout().flush().map_err(|e| e.to_string())?;

    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .map_err(|e| e.to_string())?;

    Ok(input.trim().to_string())
}
