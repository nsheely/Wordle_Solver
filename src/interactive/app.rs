//! TUI application state and logic

use crate::core::{Pattern, Word};
use crate::solver::entropy::calculate_metrics;
use crate::solver::{AdaptiveStrategy, Solver};
use anyhow::Result;
use crossterm::{
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, KeyModifiers,
    },
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{Terminal, backend::CrosstermBackend};
use std::io;

/// State snapshot for undo functionality
#[derive(Clone)]
pub struct StateSnapshot {
    pub history: Vec<HistoryEntry>,
    pub candidates_count: usize,
}

/// Application state
pub struct App<'a> {
    pub solver: Solver<'a, AdaptiveStrategy>,
    pub all_words: &'a [Word],
    pub answer_words: &'a [Word],
    pub mode: AppMode,
    pub history: Vec<HistoryEntry>,
    pub current_guess: Option<GuessInfo>,
    pub input_buffer: String,
    pub messages: Vec<Message>,
    pub stats: Statistics,
    pub should_quit: bool,
    pub input_mode: InputMode,
    pub manual_word: String,
    pub undo_stack: Vec<StateSnapshot>,
}

#[derive(Debug, Clone)]
pub enum AppMode {
    Playing,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputMode {
    Feedback,
    ManualWord,
    WinCelebration,
}

#[derive(Debug, Clone)]
pub struct HistoryEntry {
    pub guess: String,
    pub pattern: Pattern,
    pub entropy: f64,
    pub candidates_before: usize,
    pub candidates_after: usize,
}

#[derive(Debug, Clone)]
pub struct GuessInfo {
    pub word: String,
    pub entropy: f64,
    pub expected_remaining: f64,
    pub max_partition: usize,
}

#[derive(Debug, Clone)]
pub struct Message {
    pub text: String,
    pub style: MessageStyle,
}

#[derive(Debug, Clone)]
pub enum MessageStyle {
    Info,
    Success,
    Error,
}

#[derive(Debug, Default, Clone)]
pub struct Statistics {
    pub total_games: usize,
    pub games_won: usize,
    pub guess_distribution: [usize; 7],
}

impl<'a> App<'a> {
    #[must_use]
    pub fn new(all_words: &'a [Word], answer_words: &'a [Word]) -> Self {
        let solver = Solver::new(AdaptiveStrategy::default(), all_words, answer_words);

        Self {
            solver,
            all_words,
            answer_words,
            mode: AppMode::Playing,
            history: Vec::new(),
            current_guess: None,
            input_buffer: String::new(),
            messages: vec![
                Message {
                    text: "Welcome! I'll suggest optimal guesses using information theory."
                        .to_string(),
                    style: MessageStyle::Info,
                },
                Message {
                    text: "Enter feedback pattern (e.g., 'GY-GY' or '🟩🟨⬜🟩🟨')".to_string(),
                    style: MessageStyle::Info,
                },
            ],
            stats: Statistics::default(),
            should_quit: false,
            input_mode: InputMode::Feedback,
            manual_word: String::new(),
            undo_stack: Vec::new(),
        }
    }

    pub fn compute_suggestion(&mut self) {
        let guess = self.solver.next_guess(&self.get_history_for_solver());

        if let Some(guess_word) = guess {
            // Get remaining candidates for metrics
            let candidates = self.solver.get_candidates(&self.get_history_for_solver());

            // Calculate metrics
            let metrics = calculate_metrics(guess_word, &candidates);

            self.current_guess = Some(GuessInfo {
                word: guess_word.text().to_string(),
                entropy: metrics.entropy,
                expected_remaining: metrics.expected_remaining,
                max_partition: metrics.max_partition,
            });
        } else {
            self.current_guess = None;
            self.add_message("No valid guesses remaining!", MessageStyle::Error);
        }
    }

    fn get_history_for_solver(&self) -> Vec<(Word, Pattern)> {
        self.history
            .iter()
            .filter_map(|entry| Word::new(&entry.guess).ok().map(|w| (w, entry.pattern)))
            .collect()
    }

    pub fn handle_feedback(&mut self, feedback: &str) {
        // Parse the feedback pattern
        if let Some(pattern) = Pattern::from_str(feedback) {
            if let Some(guess_info) = &self.current_guess {
                let candidates_before =
                    self.solver.count_candidates(&self.get_history_for_solver());

                // Add to history
                let guess_word = guess_info.word.clone();
                self.history.push(HistoryEntry {
                    guess: guess_word,
                    pattern,
                    entropy: guess_info.entropy,
                    candidates_before,
                    candidates_after: 0, // Will be updated
                });

                // Update solver history and get new count
                let candidates_after = self.solver.count_candidates(&self.get_history_for_solver());
                if let Some(last) = self.history.last_mut() {
                    last.candidates_after = candidates_after;
                }

                // Check if solved
                if pattern.is_perfect() {
                    self.stats.games_won += 1;
                    self.stats.total_games += 1;
                    let guess_count = self.history.len();
                    if guess_count <= 6 {
                        self.stats.guess_distribution[guess_count] += 1;
                    }

                    // Switch to celebration mode
                    self.input_mode = InputMode::WinCelebration;

                    // Create celebration message based on guess count
                    let celebration = match guess_count {
                        1 => "🎯 Hole in one! Extraordinary! 🌟",
                        2 => "🔥 Magnificent! Two guesses! 🔥",
                        3 => "✨ Splendid! Three guesses! ✨",
                        4 => "👏 Great job! Four guesses! 👏",
                        5 => "🎉 Nice work! Five guesses! 🎉",
                        6 => "😅 Phew! Got it in six! 😅",
                        _ => "🎊 Solved! 🎊",
                    };

                    self.add_message(celebration, MessageStyle::Success);
                    self.add_message(
                        "Press 'n' for new game, 'u' to undo, or 'q' to quit.",
                        MessageStyle::Info,
                    );
                } else if candidates_after == 0 {
                    // Clear current guess since no valid suggestions exist
                    self.current_guess = None;
                    self.add_message(
                        "No candidates remain - pattern may be incorrect. Press 'u' to undo.",
                        MessageStyle::Error,
                    );
                } else {
                    // Compute next suggestion
                    self.compute_suggestion();
                    self.add_message(
                        &format!("{candidates_after} candidates remaining"),
                        MessageStyle::Info,
                    );
                }

                self.input_buffer.clear();
            }
        } else {
            self.add_message("Invalid pattern! Use G/Y/-  or 🟩🟨⬜", MessageStyle::Error);
        }
    }

    pub fn new_game(&mut self) {
        self.history.clear();
        self.current_guess = None;
        self.input_buffer.clear();
        self.messages.clear();
        self.input_mode = InputMode::Feedback; // Reset input mode
        self.add_message(
            "New game started! I'll suggest the optimal first guess.",
            MessageStyle::Info,
        );
        self.compute_suggestion();
    }

    pub fn undo_last(&mut self) {
        if let Some(snapshot) = self.undo_stack.pop() {
            self.history = snapshot.history;
            self.compute_suggestion();
            self.add_message("Undone!", MessageStyle::Info);
        } else if self.history.pop().is_some() {
            self.compute_suggestion();
            self.add_message("Undone!", MessageStyle::Info);
        } else {
            self.add_message("Nothing to undo!", MessageStyle::Error);
        }
    }

    pub fn add_message(&mut self, text: &str, style: MessageStyle) {
        self.messages.push(Message {
            text: text.to_string(),
            style,
        });

        // Keep only last 5 messages
        if self.messages.len() > 5 {
            self.messages.remove(0);
        }
    }

    #[must_use]
    pub fn get_candidates_count(&self) -> usize {
        self.solver.count_candidates(&self.get_history_for_solver())
    }

    pub fn use_manual_word(&mut self) {
        let word = self.manual_word.clone();

        // Validate the word exists in the allowed list
        if let Ok(word_obj) = Word::new(&word) {
            if self.all_words.iter().any(|w| w.text() == word_obj.text()) {
                // Calculate metrics for the manual word
                let candidates = self.solver.get_candidates(&self.get_history_for_solver());

                let metrics = calculate_metrics(&word_obj, &candidates);

                // Compare with suggested word if available
                if let Some(ref suggested) = self.current_guess
                    && metrics.entropy < suggested.entropy
                {
                    self.add_message(
                        &format!(
                            "Note: Suggested word had {:.2} bits ({:.2} more)",
                            suggested.entropy,
                            suggested.entropy - metrics.entropy
                        ),
                        MessageStyle::Info,
                    );
                }

                // Set the manual word as current guess
                self.current_guess = Some(GuessInfo {
                    word: word.clone(),
                    entropy: metrics.entropy,
                    expected_remaining: metrics.expected_remaining,
                    max_partition: metrics.max_partition,
                });

                self.add_message(
                    &format!(
                        "Using: {} (entropy: {:.2} bits, {:.1}x reduction)",
                        word.to_uppercase(),
                        metrics.entropy,
                        metrics.entropy.exp2()
                    ),
                    MessageStyle::Success,
                );

                // Switch back to feedback mode
                self.input_mode = InputMode::Feedback;
                self.manual_word.clear();
            } else {
                self.add_message(
                    &format!("Word '{}' not in allowed word list!", word.to_uppercase()),
                    MessageStyle::Error,
                );
            }
        } else {
            self.add_message("Invalid word format!", MessageStyle::Error);
        }
    }
}

/// Run the TUI application
///
/// # Errors
///
/// Returns an error if terminal setup/cleanup fails or if there's an I/O error
/// during rendering or event handling.
pub fn run_tui(app: App) -> Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Run app
    let res = run_app(&mut terminal, app);

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        eprintln!("Error: {err}");
    }

    Ok(())
}

fn run_app<B: ratatui::backend::Backend>(terminal: &mut Terminal<B>, mut app: App) -> Result<()> {
    // Compute initial suggestion
    app.compute_suggestion();

    loop {
        terminal.draw(|f| super::rendering::ui(f, &app))?;

        if let Event::Key(key) = event::read()? {
            // Only process key press events (fixes Windows double-input bug)
            if key.kind != KeyEventKind::Press {
                continue;
            }

            match app.input_mode {
                InputMode::WinCelebration => {
                    match key.code {
                        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            app.should_quit = true;
                        }
                        KeyCode::Char('q') => {
                            app.should_quit = true;
                        }
                        KeyCode::Char('n') => {
                            app.new_game();
                        }
                        KeyCode::Char('u') => {
                            // Allow undoing even after winning
                            app.undo_last();
                            app.input_mode = InputMode::Feedback;
                        }
                        _ => {
                            // In celebration mode, ignore other keys
                        }
                    }
                }
                InputMode::Feedback => {
                    match key.code {
                        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            app.should_quit = true;
                        }
                        KeyCode::Char('q') => {
                            app.should_quit = true;
                        }
                        KeyCode::Char('n') => {
                            app.new_game();
                            // Don't add 'n' to input buffer
                        }
                        KeyCode::Char('u') => {
                            app.undo_last();
                            // Don't add 'u' to input buffer
                        }
                        KeyCode::Tab => {
                            // Switch to manual word mode (always allow - recovery path)
                            app.input_mode = InputMode::ManualWord;
                            app.add_message("Enter your own word (5 letters)", MessageStyle::Info);
                        }
                        KeyCode::Char(c) => {
                            app.input_buffer.push(c);
                        }
                        KeyCode::Backspace => {
                            app.input_buffer.pop();
                        }
                        KeyCode::Enter => {
                            let input = app.input_buffer.clone();
                            app.handle_feedback(&input);
                        }
                        _ => {}
                    }
                }
                InputMode::ManualWord => {
                    match key.code {
                        KeyCode::Esc => {
                            app.input_mode = InputMode::Feedback;
                            app.manual_word.clear();
                            app.add_message("Cancelled manual word entry", MessageStyle::Info);
                        }
                        KeyCode::Tab => {
                            // Toggle back to feedback mode
                            app.input_mode = InputMode::Feedback;
                            app.manual_word.clear();
                        }
                        KeyCode::Char(c) if app.manual_word.len() < 5 && c.is_alphabetic() => {
                            app.manual_word.push(c.to_ascii_lowercase());
                        }
                        KeyCode::Backspace => {
                            app.manual_word.pop();
                        }
                        KeyCode::Enter => {
                            if app.manual_word.len() == 5 {
                                app.use_manual_word();
                            } else {
                                app.add_message(
                                    "Word must be exactly 5 letters!",
                                    MessageStyle::Error,
                                );
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        if app.should_quit {
            break;
        }
    }

    Ok(())
}
