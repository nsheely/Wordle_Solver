//! TUI rendering with ratatui
//!
//! Visualizations for the Wordle solver interface.

use super::app::{App, InputMode, MessageStyle};
use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Gauge, List, ListItem, Paragraph, Wrap},
};

/// Main UI rendering function
pub fn ui(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header
            Constraint::Min(10),   // Main content
            Constraint::Length(5), // Input area
            Constraint::Length(3), // Status bar
        ])
        .split(f.area());

    // Header
    render_header(f, chunks[0]);

    // Main content area - split horizontally
    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(60), // Left panel
            Constraint::Percentage(40), // Right panel
        ])
        .split(chunks[1]);

    render_main_panel(f, app, main_chunks[0]);
    render_info_panel(f, app, main_chunks[1]);

    // Input area
    render_input(f, app, chunks[2]);

    // Status bar
    render_status(f, app, chunks[3]);
}

fn render_header(f: &mut Frame, area: Rect) {
    let header = Paragraph::new("🎯 Wordle Solver - Interactive Mode")
        .style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .alignment(Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .style(Style::default().fg(Color::Cyan)),
        );
    f.render_widget(header, area);
}

fn render_main_panel(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(40), // Current guess info
            Constraint::Percentage(30), // Candidates
            Constraint::Percentage(30), // History
        ])
        .split(area);

    render_current_guess(f, app, chunks[0]);
    render_candidates(f, app, chunks[1]);
    render_history(f, app, chunks[2]);
}

fn render_current_guess(f: &mut Frame, app: &App, area: Rect) {
    if let Some(ref guess) = app.current_guess {
        // Create entropy bar (scaled to 6 bits max)
        let entropy_bar_len = (guess.entropy * 3.0).min(18.0) as usize;
        let entropy_bar =
            "█".repeat(entropy_bar_len) + &"░".repeat(18_usize.saturating_sub(entropy_bar_len));

        let content = vec![
            Line::from(vec![
                Span::raw("Suggested: "),
                Span::styled(
                    guess.word.to_uppercase(),
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(format!(
                "Entropy:   [{}] {:.3} bits",
                entropy_bar, guess.entropy
            )),
            Line::from(format!("Info gain: {:.1}x reduction", guess.entropy.exp2())),
            Line::from(format!(
                "Expected:  {:.1} candidates remain",
                guess.expected_remaining
            )),
            Line::from(format!("Worst:     {} candidates", guess.max_partition)),
        ];

        let paragraph = Paragraph::new(content)
            .block(
                Block::default()
                    .title(" Current Guess ")
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded),
            )
            .wrap(Wrap { trim: false });

        f.render_widget(paragraph, area);
    } else {
        // No current guess
        let paragraph = Paragraph::new("No suggestion available").block(
            Block::default()
                .title(" Current Guess ")
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded),
        );
        f.render_widget(paragraph, area);
    }
}

fn render_candidates(f: &mut Frame, app: &App, area: Rect) {
    let candidates_count = app.get_candidates_count();

    let content = if candidates_count == 0 {
        vec![Line::from("Game completed!")]
    } else if candidates_count <= 12 {
        // Show individual candidates (up to 12)
        let solver_history = app
            .history
            .iter()
            .filter_map(|entry| {
                crate::core::Word::new(&entry.guess)
                    .ok()
                    .map(|w| (w, entry.pattern))
            })
            .collect::<Vec<_>>();

        let candidates = app.solver.get_candidates(&solver_history);
        let candidate_refs: Vec<&crate::core::Word> = candidates.clone();

        let mut lines = vec![Line::from(vec![
            Span::raw("Remaining: "),
            Span::styled("🟢", Style::default().fg(Color::Green)),
            Span::raw(" = answer  "),
            Span::styled("⚪", Style::default().fg(Color::White)),
            Span::raw(" = guess only"),
        ])];

        for candidate in candidates.iter().take(12) {
            // Check if this word is in the answer list
            let is_answer = app
                .answer_words
                .iter()
                .any(|w| w.text() == candidate.text());

            // Calculate entropy for this candidate
            let metrics = crate::solver::entropy::calculate_metrics(candidate, &candidate_refs);

            let (prefix, style) = if is_answer {
                ("🟢", Style::default().fg(Color::Green))
            } else {
                ("⚪", Style::default().fg(Color::DarkGray))
            };

            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::raw(prefix),
                Span::raw(" "),
                Span::styled(format!("{:<5}", candidate.text().to_uppercase()), style),
                Span::styled(
                    format!(" {:.2}b", metrics.entropy),
                    Style::default().fg(Color::Cyan),
                ),
            ]));
        }
        lines
    } else {
        vec![
            Line::from(format!("{candidates_count} candidates remaining")),
            Line::from(format!(
                "Information needed: {:.2} bits",
                (candidates_count as f64).log2()
            )),
        ]
    };

    let paragraph = Paragraph::new(content).block(
        Block::default()
            .title(" Candidates ")
            .borders(Borders::ALL)
            .style(Style::default().fg(Color::Green)),
    );

    f.render_widget(paragraph, area);
}

fn render_history(f: &mut Frame, app: &App, area: Rect) {
    let history_items: Vec<ListItem> = app
        .history
        .iter()
        .rev()
        .take(5)
        .enumerate()
        .map(|(i, entry)| {
            let content = format!(
                "{}: {} {} [{:.1} bits] {} → {}",
                app.history.len() - i,
                entry.guess.to_uppercase(),
                entry.pattern.to_emoji(),
                entry.entropy,
                entry.candidates_before,
                entry.candidates_after
            );
            ListItem::new(content)
        })
        .collect();

    let history =
        List::new(history_items).block(Block::default().title(" History ").borders(Borders::ALL));

    f.render_widget(history, area);
}

fn render_info_panel(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(50), // Search space gauge
            Constraint::Percentage(50), // Messages
        ])
        .split(area);

    render_search_progress(f, app, chunks[0]);
    render_messages(f, app, chunks[1]);
}

fn render_search_progress(f: &mut Frame, app: &App, area: Rect) {
    let total_bits = 11.18; // log2(2315) - maximum entropy
    let current_candidates = app.get_candidates_count();

    // Calculate actual information gained: log2(initial) - log2(current)
    let initial_candidates = 2315_f64;
    let bits_gained = if current_candidates > 0 {
        initial_candidates.log2() - (current_candidates as f64).log2()
    } else {
        // Solved: gained all bits
        total_bits
    };

    let progress_pct = ((bits_gained / total_bits * 100.0).min(100.0)) as u16;

    let gauge = Gauge::default()
        .block(
            Block::default()
                .title(" Information Gained ")
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded),
        )
        .gauge_style(Style::default().fg(Color::Cyan))
        .percent(progress_pct)
        .label(format!(
            "{bits_gained:.1}/{total_bits:.1} bits | {current_candidates} candidates remain"
        ));

    f.render_widget(gauge, area);
}

fn render_messages(f: &mut Frame, app: &App, area: Rect) {
    let messages: Vec<ListItem> = app
        .messages
        .iter()
        .rev()
        .take(10)
        .map(|msg| {
            let style = match msg.style {
                MessageStyle::Info => Style::default().fg(Color::White),
                MessageStyle::Success => Style::default().fg(Color::Green),
                MessageStyle::Error => Style::default().fg(Color::Red),
            };
            ListItem::new(msg.text.clone()).style(style)
        })
        .collect();

    let messages_list =
        List::new(messages).block(Block::default().title(" Messages ").borders(Borders::ALL));

    f.render_widget(messages_list, area);
}

fn render_input(f: &mut Frame, app: &App, area: Rect) {
    let (title, content, color) = match app.input_mode {
        InputMode::WinCelebration => (
            " 🎉 Congratulations! | Press 'n' for new game or 'q' to quit ",
            "",
            Color::Green,
        ),
        InputMode::Feedback => (
            " Enter Feedback (G=Green Y=Yellow -=Gray, or emojis) | TAB for manual word ",
            app.input_buffer.as_str(),
            Color::Yellow,
        ),
        InputMode::ManualWord => (
            " Enter Word to Try (5 letters) | ESC to cancel ",
            app.manual_word.as_str(),
            Color::Cyan,
        ),
    };

    let input = Paragraph::new(content)
        .style(Style::default().fg(color).add_modifier(Modifier::BOLD))
        .block(
            Block::default()
                .title(title)
                .borders(Borders::ALL)
                .border_type(BorderType::Double)
                .style(Style::default().fg(color)),
        );

    f.render_widget(input, area);
}

fn render_status(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ])
        .split(area);

    let mode_text = "Mode: Playing".to_string();
    let mode = Paragraph::new(mode_text).alignment(Alignment::Center);
    f.render_widget(mode, chunks[0]);

    let stats_text = format!(
        "Games: {} | Win Rate: {:.0}%",
        app.stats.total_games,
        if app.stats.total_games > 0 {
            app.stats.games_won as f64 / app.stats.total_games as f64 * 100.0
        } else {
            0.0
        }
    );
    let stats = Paragraph::new(stats_text).alignment(Alignment::Center);
    f.render_widget(stats, chunks[1]);

    let candidates_text = format!("Candidates: {}", app.get_candidates_count());
    let candidates = Paragraph::new(candidates_text).alignment(Alignment::Center);
    f.render_widget(candidates, chunks[2]);

    let help_text = if app.get_candidates_count() == 0 && !app.history.is_empty() {
        "q: Quit | n: New Game | u: Undo"
    } else {
        "q: Quit | u: Undo | Enter: Submit | TAB: Manual Word"
    };

    let help = Paragraph::new(help_text)
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::DarkGray));
    f.render_widget(help, chunks[3]);
}
