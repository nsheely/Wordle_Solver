//! NYT Wordle API client - silent best-effort fetch of the day's solution.
//!
//! End-user binaries ship with a frozen wordlist, so when NYT introduces a
//! brand-new answer the solver would otherwise hit "word not found." This
//! module reaches out to NYT's daily endpoint at startup and merges today's
//! solution into the in-memory lists.
//!
//! Every error path returns `None` / `false`. The user must never notice this
//! ran: no logging, no panics, no surfaced failures. If the network is down,
//! NYT's schema changes, or the response is garbage, the solver falls back to
//! its embedded list as if the call never happened.

use crate::core::Word;
use std::time::Duration;

const NYT_ENDPOINT: &str = "https://www.nytimes.com/svc/wordle/v2";
const TIMEOUT: Duration = Duration::from_secs(2);

/// Fetch today's NYT Wordle solution and merge it into the given lists.
///
/// Returns `true` if anything was added. Silent on every failure.
///
/// This is the function `main` calls at startup for `play` / `simple` modes.
/// Tests should target [`merge_solution`] directly so they can fake the
/// "what NYT returned" half without hitting the network.
pub fn fetch_and_merge_today(all_words: &mut Vec<Word>, answer_words: &mut Vec<Word>) -> bool {
    fetch_today_solution().is_some_and(|s| merge_solution(&s, all_words, answer_words))
}

/// Merge a candidate solution into the guess pool and answer pool.
///
/// Idempotent: a word already present in both lists is a no-op. Invalid input
/// (anything `Word::new` rejects) is silently dropped.
///
/// Returns `true` iff at least one list grew.
pub fn merge_solution(
    solution: &str,
    all_words: &mut Vec<Word>,
    answer_words: &mut Vec<Word>,
) -> bool {
    let Ok(word) = Word::new(solution) else {
        return false;
    };
    let mut changed = false;
    if !answer_words.contains(&word) {
        answer_words.push(word);
        changed = true;
    }
    if !all_words.contains(&word) {
        all_words.push(word);
        changed = true;
    }
    changed
}

/// Fetch today's solution as a lowercase 5-letter string. `None` on any error.
fn fetch_today_solution() -> Option<String> {
    fetch_solution_for_date(&today_eastern_iso())
}

fn fetch_solution_for_date(date: &str) -> Option<String> {
    let url = format!("{NYT_ENDPOINT}/{date}.json");

    let agent = ureq::AgentBuilder::new()
        .timeout_connect(TIMEOUT)
        .timeout_read(TIMEOUT)
        .build();

    let body = agent.get(&url).call().ok()?.into_string().ok()?;
    parse_solution(&body)
}

/// Parse NYT's daily JSON, returning the solution if it's a valid 5-letter word.
fn parse_solution(body: &str) -> Option<String> {
    let json: serde_json::Value = serde_json::from_str(body).ok()?;
    let solution = json.get("solution")?.as_str()?;
    let lowered = solution.to_ascii_lowercase();
    if lowered.len() == 5 && lowered.bytes().all(|b| b.is_ascii_lowercase()) {
        Some(lowered)
    } else {
        None
    }
}

/// Today's date in NYT's reference timezone, as `YYYY-MM-DD`.
///
/// We use a fixed UTC-5 (EST) offset rather than dragging in a tz database.
/// During EDT this is off by one hour: in the brief window after midnight
/// Eastern we may query yesterday's date instead of today's. That's harmless,
/// because yesterday's word is already in the embedded list (so the merge is a
/// no-op), and the path this module exists to fix - "NYT just published a
/// brand-new answer" - is unaffected, since new answers stay current for a
/// full 24 hours.
fn today_eastern_iso() -> String {
    use chrono::{FixedOffset, Utc};
    let eastern = FixedOffset::west_opt(5 * 3600).expect("valid UTC offset");
    Utc::now()
        .with_timezone(&eastern)
        .format("%Y-%m-%d")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- parse_solution ---------------------------------------------------

    #[test]
    fn parse_solution_extracts_solution_field() {
        let body = r#"{"id":1551,"solution":"puffy","print_date":"2026-05-03"}"#;
        assert_eq!(parse_solution(body).as_deref(), Some("puffy"));
    }

    #[test]
    fn parse_solution_lowercases_uppercase_input() {
        let body = r#"{"solution":"CRANE"}"#;
        assert_eq!(parse_solution(body).as_deref(), Some("crane"));
    }

    #[test]
    fn parse_solution_rejects_wrong_length() {
        assert!(parse_solution(r#"{"solution":"cranes"}"#).is_none());
        assert!(parse_solution(r#"{"solution":"cran"}"#).is_none());
        assert!(parse_solution(r#"{"solution":""}"#).is_none());
    }

    #[test]
    fn parse_solution_rejects_non_alpha() {
        assert!(parse_solution(r#"{"solution":"cr4ne"}"#).is_none());
        assert!(parse_solution(r#"{"solution":"cran-"}"#).is_none());
    }

    #[test]
    fn parse_solution_handles_missing_field() {
        let body = r#"{"id":1551,"print_date":"2026-05-03"}"#;
        assert!(parse_solution(body).is_none());
    }

    #[test]
    fn parse_solution_handles_wrong_type() {
        // NYT could theoretically send `solution: null` or a number. Don't blow up.
        assert!(parse_solution(r#"{"solution":null}"#).is_none());
        assert!(parse_solution(r#"{"solution":42}"#).is_none());
    }

    #[test]
    fn parse_solution_handles_malformed_json() {
        assert!(parse_solution("not json").is_none());
        assert!(parse_solution("").is_none());
        assert!(parse_solution("{").is_none());
    }

    // --- merge_solution ---------------------------------------------------

    fn words(strs: &[&str]) -> Vec<Word> {
        strs.iter().map(|s| Word::new(*s).unwrap()).collect()
    }

    #[test]
    fn merge_adds_new_word_to_both_lists() {
        // "zzzzz" is a syntactically valid Word that won't be in either list.
        // Stand-in for "NYT just shipped a brand-new answer."
        let mut all = words(&["crane", "slate"]);
        let mut answers = words(&["crane"]);

        let changed = merge_solution("zzzzz", &mut all, &mut answers);

        assert!(changed);
        assert_eq!(all.len(), 3);
        assert_eq!(answers.len(), 2);
        assert!(all.contains(&Word::new("zzzzz").unwrap()));
        assert!(answers.contains(&Word::new("zzzzz").unwrap()));
    }

    #[test]
    fn merge_is_noop_when_word_in_both_lists() {
        let mut all = words(&["crane", "slate"]);
        let mut answers = words(&["crane"]);

        let changed = merge_solution("crane", &mut all, &mut answers);

        assert!(!changed);
        assert_eq!(all, words(&["crane", "slate"]));
        assert_eq!(answers, words(&["crane"]));
    }

    #[test]
    fn merge_adds_to_answers_only_when_already_in_allowed() {
        // Defensive: if a word is in the guess pool but not the answer pool
        // (the historical state of pre-NYT custom dictionaries), top it up.
        let mut all = words(&["crane", "slate"]);
        let mut answers = words(&["crane"]);

        let changed = merge_solution("slate", &mut all, &mut answers);

        assert!(changed);
        assert_eq!(all.len(), 2, "allowed list shouldn't grow");
        assert_eq!(answers, words(&["crane", "slate"]));
    }

    #[test]
    fn merge_uppercases_input_via_word_constructor() {
        // Word::new lowercases for us, so the merge accepts whatever case NYT sends.
        let mut all = words(&["crane"]);
        let mut answers = words(&["crane"]);

        assert!(merge_solution("PUFFY", &mut all, &mut answers));
        assert!(all.contains(&Word::new("puffy").unwrap()));
    }

    #[test]
    fn merge_rejects_invalid_words_silently() {
        let mut all = words(&["crane"]);
        let mut answers = words(&["crane"]);

        assert!(!merge_solution("toolong", &mut all, &mut answers));
        assert!(!merge_solution("abc", &mut all, &mut answers));
        assert!(!merge_solution("", &mut all, &mut answers));
        assert!(!merge_solution("cr4ne", &mut all, &mut answers));

        assert_eq!(all, words(&["crane"]));
        assert_eq!(answers, words(&["crane"]));
    }

    #[test]
    fn merge_after_parse_simulates_full_pipeline() {
        // End-to-end without the HTTP: feed a fake NYT JSON response through
        // the parser, hand the result to the merger. This exercises the same
        // chain `fetch_and_merge_today` runs once the bytes are off the wire.
        let fake_response = r#"{"id":9999,"solution":"zzzzz","print_date":"2099-12-31"}"#;

        let mut all = words(&["crane", "slate"]);
        let mut answers = words(&["crane"]);

        let solution = parse_solution(fake_response).expect("valid response");
        let changed = merge_solution(&solution, &mut all, &mut answers);

        assert!(changed);
        assert!(answers.iter().any(|w| w.text() == "zzzzz"));
        assert!(all.iter().any(|w| w.text() == "zzzzz"));
    }

    #[test]
    fn merge_after_parse_skips_garbage_response() {
        let mut all = words(&["crane"]);
        let mut answers = words(&["crane"]);
        let before_all = all.clone();
        let before_answers = answers.clone();

        // Garbage from upstream → parse returns None → no merge happens.
        if let Some(s) = parse_solution("upstream had a bad day") {
            merge_solution(&s, &mut all, &mut answers);
        }

        assert_eq!(all, before_all);
        assert_eq!(answers, before_answers);
    }

    // --- today_eastern_iso ------------------------------------------------

    #[test]
    fn today_eastern_iso_is_well_formed() {
        let s = today_eastern_iso();
        assert_eq!(s.len(), 10, "expected YYYY-MM-DD, got {s:?}");
        assert_eq!(s.as_bytes()[4], b'-');
        assert_eq!(s.as_bytes()[7], b'-');
        assert!(s[..4].chars().all(|c| c.is_ascii_digit()));
        assert!(s[5..7].chars().all(|c| c.is_ascii_digit()));
        assert!(s[8..].chars().all(|c| c.is_ascii_digit()));
    }
}
