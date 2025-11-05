//! Wordle word representation
//!
//! A Word stores a 5-letter word for efficient pattern calculation.

use std::fmt;

/// A 5-letter Wordle word
///
/// Stores as byte array; text is reconstructed on-demand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Word {
    chars: [u8; 5],
}

/// Error type for invalid words
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WordError {
    InvalidLength(usize),
    NonAscii,
    InvalidCharacters,
}

impl fmt::Display for WordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength(len) => {
                write!(f, "Word must be exactly 5 letters, got {len}")
            }
            Self::NonAscii => write!(f, "Word must contain only ASCII letters"),
            Self::InvalidCharacters => write!(f, "Word contains invalid characters"),
        }
    }
}

impl std::error::Error for WordError {}

impl Word {
    /// Create a new Word from a string
    ///
    /// Converts the input to lowercase and validates it meets Wordle requirements.
    ///
    /// # Errors
    /// Returns `WordError` if:
    /// - Length is not exactly 5
    /// - Contains non-ASCII characters
    /// - Contains non-alphabetic characters
    ///
    /// # Examples
    /// ```
    /// use wordle_solver::core::Word;
    ///
    /// let word = Word::new("crane").unwrap();
    /// assert_eq!(word.text(), "crane");
    ///
    /// assert!(Word::new("too long").is_err());
    /// assert!(Word::new("sh0rt").is_err());
    /// ```
    ///
    /// # Panics
    /// Never panics. The `expect` call is guaranteed safe by prior length validation.
    pub fn new(text: impl Into<String>) -> Result<Self, WordError> {
        let text: String = text.into().to_lowercase();

        // Validate length
        if text.len() != 5 {
            return Err(WordError::InvalidLength(text.len()));
        }

        // Validate ASCII and alphabetic
        if !text.is_ascii() {
            return Err(WordError::NonAscii);
        }

        if !text.chars().all(|c| c.is_ascii_lowercase()) {
            return Err(WordError::InvalidCharacters);
        }

        // Convert to bytes - safe to unwrap as we validated length == 5
        let chars: [u8; 5] = text
            .as_bytes()
            .try_into()
            .expect("length already validated");

        Ok(Self { chars })
    }

    /// Get the word as a string slice
    ///
    /// Reconstructs the string from the byte array on-demand.
    /// This is safe because Word validates ASCII lowercase at construction.
    ///
    /// # Panics
    /// Never panics. The `expect` call is guaranteed safe because Word validates
    /// ASCII lowercase characters at construction time.
    #[inline]
    #[must_use]
    pub fn text(&self) -> &str {
        std::str::from_utf8(&self.chars).expect("Word chars are always valid UTF-8")
    }

    /// Get the word as a byte array
    #[inline]
    #[must_use]
    pub const fn chars(&self) -> &[u8; 5] {
        &self.chars
    }

    /// Get the character at a specific position (0-4)
    ///
    /// # Panics
    /// Panics if position >= 5
    #[inline]
    #[must_use]
    pub const fn char_at(&self, position: usize) -> u8 {
        self.chars[position]
    }

    /// Check if the word contains a specific letter
    #[inline]
    #[must_use]
    pub fn has_letter(&self, letter: u8) -> bool {
        self.chars.contains(&letter)
    }

    /// Get all positions where a letter appears
    ///
    /// Returns a Vec of positions. Empty if the letter doesn't appear.
    #[must_use]
    pub fn positions_of(&self, letter: u8) -> Vec<usize> {
        self.chars
            .iter()
            .enumerate()
            .filter_map(|(i, &ch)| if ch == letter { Some(i) } else { None })
            .collect()
    }

    /// Get the count of each letter in the word
    ///
    /// Returns array where index represents the letter (a=0, b=1, ..., z=25)
    /// and the value is the count of that letter in the word.
    #[inline]
    pub(crate) fn char_counts(self) -> [u8; 26] {
        let mut counts = [0u8; 26];
        for &ch in &self.chars {
            counts[(ch - b'a') as usize] += 1;
        }
        counts
    }
}

impl fmt::Display for Word {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.text())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_creation_valid() {
        let word = Word::new("crane").unwrap();
        assert_eq!(word.text(), "crane");
        assert_eq!(word.chars(), b"crane");
    }

    #[test]
    fn word_creation_uppercase_normalized() {
        let word = Word::new("CRANE").unwrap();
        assert_eq!(word.text(), "crane");

        let word2 = Word::new("CrAnE").unwrap();
        assert_eq!(word2.text(), "crane");
    }

    #[test]
    fn word_creation_invalid_length() {
        assert!(matches!(
            Word::new("too long"),
            Err(WordError::InvalidLength(8))
        ));
        assert!(matches!(
            Word::new("shrt"),
            Err(WordError::InvalidLength(4))
        ));
        assert!(matches!(Word::new(""), Err(WordError::InvalidLength(0))));
    }

    #[test]
    fn word_creation_invalid_characters() {
        assert!(Word::new("cran3").is_err()); // Number
        assert!(Word::new("cran ").is_err()); // Space
        assert!(Word::new("cran!").is_err()); // Punctuation
    }

    #[test]
    fn word_char_at() {
        let word = Word::new("crane").unwrap();
        assert_eq!(word.char_at(0), b'c');
        assert_eq!(word.char_at(1), b'r');
        assert_eq!(word.char_at(2), b'a');
        assert_eq!(word.char_at(3), b'n');
        assert_eq!(word.char_at(4), b'e');
    }

    #[test]
    fn word_has_letter() {
        let word = Word::new("crane").unwrap();
        assert!(word.has_letter(b'c'));
        assert!(word.has_letter(b'r'));
        assert!(word.has_letter(b'a'));
        assert!(!word.has_letter(b'z'));
        assert!(!word.has_letter(b'x'));
    }

    #[test]
    fn word_positions_of() {
        let word = Word::new("crane").unwrap();
        assert_eq!(word.positions_of(b'c'), &[0]);
        assert_eq!(word.positions_of(b'r'), &[1]);
        assert_eq!(word.positions_of(b'a'), &[2]);
        assert_eq!(word.positions_of(b'z'), &[]);
    }

    #[test]
    fn word_positions_of_duplicates() {
        let word = Word::new("speed").unwrap();
        assert_eq!(word.positions_of(b'e'), &[2, 3]); // Both E positions
        assert_eq!(word.positions_of(b's'), &[0]);
        assert_eq!(word.positions_of(b'p'), &[1]);
        assert_eq!(word.positions_of(b'd'), &[4]);
    }

    #[test]
    fn word_positions_of_all_same() {
        let word = Word::new("aaaaa").unwrap();
        assert_eq!(word.positions_of(b'a'), &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn word_char_counts() {
        let word = Word::new("speed").unwrap();
        let counts = word.char_counts();
        assert_eq!(counts[(b's' - b'a') as usize], 1);
        assert_eq!(counts[(b'p' - b'a') as usize], 1);
        assert_eq!(counts[(b'e' - b'a') as usize], 2);
        assert_eq!(counts[(b'd' - b'a') as usize], 1);
    }

    #[test]
    fn word_char_counts_all_unique() {
        let word = Word::new("crane").unwrap();
        let counts = word.char_counts();
        let non_zero = counts.iter().filter(|&&c| c > 0).count();
        assert_eq!(non_zero, 5);
        assert!(counts.iter().filter(|&&c| c > 0).all(|&count| count == 1));
    }

    #[test]
    fn word_char_counts_all_same() {
        let word = Word::new("aaaaa").unwrap();
        let counts = word.char_counts();
        let non_zero = counts.iter().filter(|&&c| c > 0).count();
        assert_eq!(non_zero, 1);
        assert_eq!(counts[(b'a' - b'a') as usize], 5);
    }

    #[test]
    fn word_display() {
        let word = Word::new("crane").unwrap();
        assert_eq!(format!("{word}"), "crane");
    }

    #[test]
    fn word_equality() {
        let word1 = Word::new("crane").unwrap();
        let word2 = Word::new("crane").unwrap();
        let word3 = Word::new("CRANE").unwrap();
        let word4 = Word::new("slate").unwrap();

        assert_eq!(word1, word2);
        assert_eq!(word1, word3); // Case insensitive
        assert_ne!(word1, word4);
    }
}
