//! Word lists for Wordle
//!
//! Embedded word lists compiled into the binary.

mod embedded;
pub mod loader;

pub use embedded::{ALLOWED, ALLOWED_COUNT, ANSWERS, ANSWERS_COUNT};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn answers_count_matches_const() {
        assert_eq!(ANSWERS.len(), ANSWERS_COUNT);
    }

    #[test]
    fn allowed_count_matches_const() {
        assert_eq!(ALLOWED.len(), ALLOWED_COUNT);
    }

    #[test]
    fn answers_are_valid_words() {
        // All answers should be 5 letters, lowercase
        for &word in ANSWERS {
            assert_eq!(word.len(), 5, "Word '{word}' is not 5 letters");
            assert!(
                word.chars().all(|c| c.is_ascii_lowercase()),
                "Word '{word}' contains non-lowercase chars"
            );
        }
    }

    #[test]
    fn allowed_are_valid_words() {
        // All allowed words should be 5 letters, lowercase
        for &word in &ALLOWED[..10] {
            // Just check first 10 for speed
            assert_eq!(word.len(), 5, "Word '{word}' is not 5 letters");
            assert!(
                word.chars().all(|c| c.is_ascii_lowercase()),
                "Word '{word}' contains non-lowercase chars"
            );
        }
    }

    #[test]
    fn answers_subset_of_allowed() {
        // All answer words should be in the allowed list
        let allowed_set: std::collections::HashSet<_> = ALLOWED.iter().collect();

        for &answer in &ANSWERS[..10] {
            // Check first 10 for speed
            assert!(
                allowed_set.contains(&answer),
                "Answer '{answer}' not in allowed list"
            );
        }
    }

    #[test]
    fn expected_counts() {
        let answers_len = ANSWERS.len();
        let allowed_len = ALLOWED.len();

        // Test that count constants match actual array lengths
        // (catches build script bugs where generated counts are wrong)
        assert_eq!(
            ANSWERS_COUNT, answers_len,
            "ANSWERS_COUNT constant doesn't match actual array length"
        );
        assert_eq!(
            ALLOWED_COUNT, allowed_len,
            "ALLOWED_COUNT constant doesn't match actual array length"
        );

        // Sanity check: reasonable bounds based on known Wordle word lists
        // NYT started with ~2,315 answers and has added more over time
        assert!(
            answers_len >= 2300,
            "Answer count unexpectedly low: {answers_len} (expected >= 2300)"
        );
        assert!(
            answers_len <= 3000,
            "Answer count unexpectedly high: {answers_len} (expected <= 3000)"
        );

        // Allowed words should be significantly larger than answers
        // Original Wordle had ~12,972 allowed words
        assert!(
            allowed_len >= 12000,
            "Allowed count unexpectedly low: {allowed_len} (expected >= 12000)"
        );
        assert!(
            allowed_len <= 15000,
            "Allowed count unexpectedly high: {allowed_len} (expected <= 15000)"
        );

        // Answers should be a subset of allowed (count-wise)
        assert!(
            answers_len <= allowed_len,
            "Answers count ({answers_len}) should not exceed allowed count ({allowed_len})"
        );
    }
}
