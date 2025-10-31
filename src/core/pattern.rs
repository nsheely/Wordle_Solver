//! Wordle feedback pattern calculation and representation
//!
//! A pattern encodes the feedback from a guess using base-3 encoding:
//! - 0 = Gray (letter not in word)
//! - 1 = Yellow (letter in word, wrong position)
//! - 2 = Green (letter in correct position)
//!
//! The pattern is stored as a single u8 value (0-242), where each position
//! contributes digit × 3^position to the total.

use super::Word;

/// Feedback pattern for a Wordle guess
///
/// Represents the colored feedback as a single byte value.
/// Value range: 0-242 (3^5 - 1 = 243 possible patterns)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Pattern(u8);

impl Pattern {
    /// All greens (perfect match)
    pub const PERFECT: Self = Self(242); // 2 + 2×3 + 2×9 + 2×27 + 2×81

    /// Create a new pattern from a raw value
    ///
    /// # Panics
    /// Panics in debug mode if value >= 243
    #[inline]
    #[must_use]
    pub const fn new(value: u8) -> Self {
        debug_assert!(value < 243, "Pattern value must be < 243");
        Self(value)
    }

    /// Get the raw pattern value (0-242)
    #[inline]
    #[must_use]
    pub const fn value(self) -> u8 {
        self.0
    }

    /// Check if this is a perfect match (all greens)
    #[inline]
    #[must_use]
    pub const fn is_perfect(self) -> bool {
        self.0 == 242
    }

    /// Calculate the pattern when `guess` is guessed and `answer` is the target
    ///
    /// This implements Wordle's exact feedback rules, including proper handling
    /// of duplicate letters.
    ///
    /// # Algorithm
    /// 1. First pass: Mark all exact matches (greens) and remove from available pool
    /// 2. Second pass: Mark present-but-wrong-position (yellows) from remaining pool
    /// 3. Encode as base-3 number
    ///
    /// # Examples
    /// ```
    /// use wordle_solver::core::{Word, Pattern};
    ///
    /// let guess = Word::new("crane").unwrap();
    /// let answer = Word::new("slate").unwrap();
    /// let pattern = Pattern::calculate(&guess, &answer);
    ///
    /// // C(gray) R(yellow) A(green) N(gray) E(green)
    /// // 0 + 1×3 + 2×9 + 0×27 + 2×81 = 180
    /// assert_eq!(pattern.value(), 180);
    /// ```
    #[inline]
    #[must_use]
    pub fn calculate(guess: &Word, answer: &Word) -> Self {
        let mut result = [0u8; 5];
        let mut answer_available = answer.char_counts();

        // First pass: Mark greens (exact position matches)
        #[allow(clippy::needless_range_loop)] // Need index for positional matching
        for i in 0..5 {
            if guess.chars()[i] == answer.chars()[i] {
                result[i] = 2; // Green

                // Remove from available pool for yellow detection
                let idx = (guess.chars()[i] - b'a') as usize;
                answer_available[idx] -= 1; // Safe: letter must exist in answer
            }
        }

        // Second pass: Mark yellows (letter exists but wrong position)
        #[allow(clippy::needless_range_loop)] // Need index to update result array
        for i in 0..5 {
            if result[i] == 0 {
                let idx = (guess.chars()[i] - b'a') as usize;
                if answer_available[idx] > 0 {
                    result[i] = 1; // Yellow
                    answer_available[idx] -= 1;
                }
            }
        }

        // Encode as base-3 number: digit[0] + digit[1]*3 + digit[2]*9 + digit[3]*27 + digit[4]*81
        let pattern = result[0] + result[1] * 3 + result[2] * 9 + result[3] * 27 + result[4] * 81;

        Self(pattern)
    }

    /// Count the number of green feedback squares
    #[must_use]
    pub fn count_greens(self) -> u8 {
        let mut count = 0;
        let mut val = self.0;

        for _ in 0..5 {
            if val % 3 == 2 {
                count += 1;
            }
            val /= 3;
        }

        count
    }

    /// Count the number of yellow feedback squares
    #[must_use]
    pub fn count_yellows(self) -> u8 {
        let mut count = 0;
        let mut val = self.0;

        for _ in 0..5 {
            if val % 3 == 1 {
                count += 1;
            }
            val /= 3;
        }

        count
    }

    /// Parse a pattern from a string like "GYGGY" or "🟩🟨🟩🟩🟨"
    ///
    /// Accepts:
    /// - 'G'/'g'/🟩 for green
    /// - 'Y'/'y'/🟨 for yellow
    /// - '-'/'_'/⬜ for gray
    ///
    /// # Examples
    /// ```
    /// use wordle_solver::core::Pattern;
    ///
    /// let p1 = Pattern::from_str("GY-GY").unwrap();
    /// let p2 = Pattern::from_str("🟩🟨⬜🟩🟨").unwrap();
    /// assert_eq!(p1, p2);
    /// ```
    #[must_use]
    #[allow(clippy::should_implement_trait)] // Provides ergonomic Option API; FromStr trait also implemented below
    pub fn from_str(s: &str) -> Option<Self> {
        let chars: Vec<char> = s.chars().collect();

        if chars.len() != 5 {
            return None;
        }

        let mut pattern = 0u8;
        let mut multiplier = 1u8;

        for ch in chars {
            let digit = match ch {
                'G' | 'g' | '🟩' => 2,
                'Y' | 'y' | '🟨' => 1,
                '-' | '_' | '⬜' => 0,
                _ => return None,
            };
            pattern += digit * multiplier;
            multiplier *= 3;
        }

        Some(Self(pattern))
    }

    /// Convert pattern to emoji string
    ///
    /// Returns a string like "🟩🟨⬜🟩🟨" representing the pattern.
    ///
    /// # Examples
    /// ```
    /// use wordle_solver::core::Pattern;
    ///
    /// let p = Pattern::from_str("GY-GY").unwrap();
    /// assert_eq!(p.to_emoji(), "🟩🟨⬜🟩🟨");
    /// ```
    #[must_use]
    pub fn to_emoji(self) -> String {
        let mut result = String::with_capacity(10); // 2 bytes per emoji
        let mut val = self.0;

        for _ in 0..5 {
            let digit = val % 3;
            result.push(match digit {
                2 => '🟩', // Green
                1 => '🟨', // Yellow
                _ => '⬜', // Gray
            });
            val /= 3;
        }

        result
    }
}

impl std::str::FromStr for Pattern {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_str(s).ok_or_else(|| format!("Invalid pattern string: {s}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pattern_perfect_constant() {
        assert_eq!(Pattern::PERFECT.value(), 242);
        assert!(Pattern::PERFECT.is_perfect());
        assert_eq!(Pattern::PERFECT.count_greens(), 5);
        assert_eq!(Pattern::PERFECT.count_yellows(), 0);
    }

    #[test]
    fn pattern_all_gray() {
        let guess = Word::new("abcde").unwrap();
        let answer = Word::new("fghij").unwrap();
        let pattern = Pattern::calculate(&guess, &answer);

        assert_eq!(pattern.value(), 0);
        assert_eq!(pattern.count_greens(), 0);
        assert_eq!(pattern.count_yellows(), 0);
    }

    #[test]
    fn pattern_all_green() {
        let word = Word::new("crane").unwrap();
        let pattern = Pattern::calculate(&word, &word);

        assert_eq!(pattern, Pattern::PERFECT);
        assert_eq!(pattern.count_greens(), 5);
    }

    #[test]
    fn pattern_duplicate_letters_green_takes_priority() {
        // SPEED vs ERASE
        // S(yellow) P(gray) E(yellow) E(yellow) D(gray)
        // S is at position 3 in ERASE, so yellow
        // Both E's are yellow (ERASE has 2 E's at positions 0 and 4)
        let guess = Word::new("speed").unwrap();
        let answer = Word::new("erase").unwrap();
        let pattern = Pattern::calculate(&guess, &answer);

        // S(yellow)=1, P(gray)=0, E(yellow)=1, E(yellow)=1, D(gray)=0
        // 1 + 0×3 + 1×9 + 1×27 + 0×81 = 37
        assert_eq!(pattern.value(), 37);
        assert_eq!(pattern.count_greens(), 0);
        assert_eq!(pattern.count_yellows(), 3);
    }

    #[test]
    fn pattern_duplicate_letters_complex() {
        // ROBOT vs FLOOR
        // R(yellow) O(yellow) B(gray) O(green) T(gray)
        // First O is yellow (wrong position), second O is green (correct position)
        let guess = Word::new("robot").unwrap();
        let answer = Word::new("floor").unwrap();
        let pattern = Pattern::calculate(&guess, &answer);

        // R(yellow)=1, O(yellow)=1, B(gray)=0, O(green)=2, T(gray)=0
        // 1 + 1×3 + 0×9 + 2×27 + 0×81 = 58
        assert_eq!(pattern.value(), 58);
        assert_eq!(pattern.count_greens(), 1);
        assert_eq!(pattern.count_yellows(), 2);
    }

    #[test]
    fn pattern_from_str_valid() {
        let p1 = Pattern::from_str("GYG--").unwrap();
        let p2 = Pattern::from_str("🟩🟨🟩⬜⬜").unwrap();
        let p3 = Pattern::from_str("gyg__").unwrap();

        assert_eq!(p1, p2);
        assert_eq!(p1, p3);

        // G=2, Y=1, G=2, -=0, -=0
        // 2 + 1×3 + 2×9 + 0×27 + 0×81 = 23
        assert_eq!(p1.value(), 23);
    }

    #[test]
    fn pattern_from_str_invalid() {
        assert!(Pattern::from_str("GYGGYX").is_none()); // Too long (6 chars)
        assert!(Pattern::from_str("GYG").is_none()); // Too short
        assert!(Pattern::from_str("GXGGY").is_none()); // Invalid char
        assert!(Pattern::from_str("").is_none()); // Empty
    }

    #[test]
    fn pattern_count_feedback() {
        // Create pattern manually: YGGYY
        // Y=1, G=2, G=2, Y=1, Y=1
        // 1 + 2×3 + 2×9 + 1×27 + 1×81 = 133
        let pattern = Pattern::new(133);

        assert_eq!(pattern.count_greens(), 2);
        assert_eq!(pattern.count_yellows(), 3);
    }

    #[test]
    fn pattern_symmetry() {
        // Pattern of word vs itself is always perfect
        for word in ["crane", "slate", "audio", "zzzzz", "aaaaa"] {
            let w = Word::new(word).unwrap();
            assert_eq!(Pattern::calculate(&w, &w), Pattern::PERFECT);
        }
    }

    #[test]
    fn pattern_real_wordle_example() {
        // Classic Wordle example: CRANE vs SLATE
        let guess = Word::new("crane").unwrap();
        let answer = Word::new("slate").unwrap();
        let pattern = Pattern::calculate(&guess, &answer);

        // C(gray)=0, R(gray)=0, A(green)=2, N(gray)=0, E(green)=2
        // R is gray because SLATE has no R
        // 0 + 0×3 + 2×9 + 0×27 + 2×81 = 180
        assert_eq!(pattern.value(), 180);
        assert_eq!(pattern.count_greens(), 2); // A and E
        assert_eq!(pattern.count_yellows(), 0); // No yellows
    }
}
