//! Entropy calculation for Wordle patterns
//!
//! Calculates Shannon entropy to measure the information gain from a guess.
//! Higher entropy means the guess better splits the remaining candidates.

use crate::core::{Pattern, Word};

/// Metrics for evaluating a guess
#[derive(Debug, Clone, Copy)]
pub struct GuessMetrics {
    /// Entropy (expected information gain in bits)
    pub entropy: f64,
    /// Expected number of remaining candidates after this guess
    pub expected_remaining: f64,
    /// Maximum partition size (worst-case remaining candidates)
    pub max_partition: usize,
}

/// Calculate Shannon entropy for a guess against candidates
///
/// Returns the expected information gain in bits.
///
/// # Formula
/// H(X) = -Σ p(x) * log₂(p(x))
///
/// where p(x) is the probability of observing pattern x.
///
/// # Examples
/// ```
/// use wordle_solver::core::Word;
/// use wordle_solver::solver::entropy::calculate_entropy;
///
/// let guess = Word::new("crane").unwrap();
/// let candidates = vec![
///     Word::new("slate").unwrap(),
///     Word::new("irate").unwrap(),
/// ];
/// let candidate_refs: Vec<&Word> = candidates.iter().collect();
///
/// let entropy = calculate_entropy(&guess, &candidate_refs);
/// assert!(entropy > 0.0 && entropy <= 1.0); // log2(2) = 1 bit max
/// ```
#[must_use]
pub fn calculate_entropy(guess: &Word, candidates: &[&Word]) -> f64 {
    if candidates.is_empty() {
        return 0.0;
    }

    // Group candidates by pattern
    let pattern_counts = group_by_pattern(*guess, candidates);

    // Calculate Shannon entropy from array
    shannon_entropy_array(&pattern_counts, candidates.len())
}

/// Calculate Shannon entropy from array of pattern counts
///
/// Internal optimized version that works with fixed-size arrays.
#[inline]
fn shannon_entropy_array(pattern_counts: &[usize; 243], total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }

    let total_f = total as f64;
    pattern_counts
        .iter()
        .filter(|&&count| count > 0)
        .map(|&count| {
            let p = count as f64 / total_f;
            -p * p.log2()
        })
        .sum()
}

/// Group candidates by the pattern they produce with the guess
fn group_by_pattern(guess: Word, candidates: &[&Word]) -> [usize; 243] {
    let mut counts = [0usize; 243]; // Array for all 243 possible patterns

    for &candidate in candidates {
        let pattern = Pattern::calculate(&guess, candidate);
        counts[pattern.value() as usize] += 1;
    }

    counts
}

/// Calculate Shannon entropy from pattern distribution
///
/// H = -Σ p * log₂(p)
///
/// # Examples
/// ```
/// use wordle_solver::solver::entropy::shannon_entropy;
/// use std::collections::HashMap;
/// use wordle_solver::core::Pattern;
///
/// let mut uniform = HashMap::new();
/// uniform.insert(Pattern::new(0), 25);
/// uniform.insert(Pattern::new(1), 25);
/// uniform.insert(Pattern::new(2), 25);
/// uniform.insert(Pattern::new(3), 25);
///
/// let entropy = shannon_entropy(&uniform);
/// assert!((entropy - 2.0).abs() < 0.001); // log2(4) = 2 bits
/// ```
#[must_use]
pub fn shannon_entropy<S>(pattern_counts: &std::collections::HashMap<Pattern, usize, S>) -> f64
where
    S: std::hash::BuildHasher,
{
    let total = pattern_counts.values().sum::<usize>() as f64;

    if total == 0.0 {
        return 0.0;
    }

    pattern_counts
        .values()
        .filter(|&&count| count > 0)
        .map(|&count| {
            let p = count as f64 / total;
            -p * p.log2()
        })
        .sum()
}

/// Calculate comprehensive metrics for a guess
///
/// Returns entropy, expected remaining candidates, and max partition size.
/// This enables sophisticated tiebreaking strategies.
#[must_use]
pub fn calculate_metrics(guess: &Word, candidates: &[&Word]) -> GuessMetrics {
    if candidates.is_empty() {
        return GuessMetrics {
            entropy: 0.0,
            expected_remaining: 0.0,
            max_partition: 0,
        };
    }

    // Count how many candidates produce each pattern
    let pattern_counts = group_by_pattern(*guess, candidates);

    let total = candidates.len() as f64;
    let mut entropy = 0.0;
    let mut expected_remaining = 0.0;
    let mut max_partition = 0;

    // Single pass through pattern counts to calculate all metrics
    for &count in &pattern_counts {
        if count > 0 {
            let p = count as f64 / total;
            entropy += -p * p.log2();
            expected_remaining += p * count as f64;
            max_partition = max_partition.max(count);
        }
    }

    GuessMetrics {
        entropy,
        expected_remaining,
        max_partition,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn shannon_entropy_uniform_distribution() {
        // 4 patterns, each appears once = log2(4) = 2 bits
        let mut counts = HashMap::new();
        counts.insert(Pattern::new(0), 1);
        counts.insert(Pattern::new(1), 1);
        counts.insert(Pattern::new(2), 1);
        counts.insert(Pattern::new(3), 1);

        let entropy = shannon_entropy(&counts);
        assert!((entropy - 2.0).abs() < 0.001);
    }

    #[test]
    fn shannon_entropy_certain_outcome() {
        // Only one pattern = 0 bits (no uncertainty)
        let mut counts = HashMap::new();
        counts.insert(Pattern::new(0), 10);

        let entropy = shannon_entropy(&counts);
        assert!(entropy.abs() < 0.001);
    }

    #[test]
    fn shannon_entropy_skewed_distribution() {
        // Skewed distribution has less entropy than uniform
        let mut uniform = HashMap::new();
        uniform.insert(Pattern::new(0), 25);
        uniform.insert(Pattern::new(1), 25);
        uniform.insert(Pattern::new(2), 25);
        uniform.insert(Pattern::new(3), 25);

        let mut skewed = HashMap::new();
        skewed.insert(Pattern::new(0), 97);
        skewed.insert(Pattern::new(1), 1);
        skewed.insert(Pattern::new(2), 1);
        skewed.insert(Pattern::new(3), 1);

        assert!(shannon_entropy(&uniform) > shannon_entropy(&skewed));
    }

    #[test]
    fn shannon_entropy_bounds() {
        // Entropy is always non-negative and bounded by log2(n)
        let mut counts = HashMap::new();
        counts.insert(Pattern::new(0), 10);
        counts.insert(Pattern::new(1), 20);
        counts.insert(Pattern::new(2), 30);

        let entropy = shannon_entropy(&counts);
        assert!(entropy >= 0.0);
        assert!(entropy <= (counts.len() as f64).log2());
    }

    #[test]
    fn shannon_entropy_empty() {
        let counts: HashMap<Pattern, usize> = HashMap::new();
        let entropy = shannon_entropy(&counts);
        assert!((entropy - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn shannon_entropy_with_zero_counts() {
        // HashMap with both zero and non-zero counts
        // Zero counts should be filtered out
        let mut counts = HashMap::new();
        counts.insert(Pattern::new(0), 0); // Zero count
        counts.insert(Pattern::new(1), 10); // Non-zero
        counts.insert(Pattern::new(2), 0); // Zero count
        counts.insert(Pattern::new(3), 10); // Non-zero

        let entropy = shannon_entropy(&counts);

        // Only non-zero counts contribute
        // Two outcomes with equal probability (10 each) = 1 bit
        assert!((entropy - 1.0).abs() < 0.001);
    }

    #[test]
    fn calculate_entropy_real_words() {
        let guess = Word::new("crane").unwrap();
        let candidates = [
            Word::new("slate").unwrap(),
            Word::new("irate").unwrap(),
            Word::new("trace").unwrap(),
            Word::new("raise").unwrap(),
        ];
        let candidate_refs: Vec<&Word> = candidates.iter().collect();

        let entropy = calculate_entropy(&guess, &candidate_refs);

        // With 4 candidates and good diversity, expect 1.5-2.0 bits
        assert!(entropy > 1.0 && entropy <= 2.0);
    }

    #[test]
    fn calculate_entropy_all_same_pattern() {
        // If all candidates produce same pattern, entropy = 0
        let guess = Word::new("zzzzz").unwrap();
        let candidates = [
            Word::new("aaaaa").unwrap(),
            Word::new("bbbbb").unwrap(),
            Word::new("ccccc").unwrap(),
        ];
        let candidate_refs: Vec<&Word> = candidates.iter().collect();

        let entropy = calculate_entropy(&guess, &candidate_refs);

        // All produce same pattern (all gray) = 0 bits
        assert!(entropy.abs() < 0.001);
    }

    #[test]
    fn calculate_entropy_perfect_split() {
        // Perfect binary split = 1 bit
        let guess = Word::new("slate").unwrap();
        let candidates = [
            Word::new("slate").unwrap(), // Perfect match
            Word::new("zzzzz").unwrap(), // No match
        ];
        let candidate_refs: Vec<&Word> = candidates.iter().collect();

        let entropy = calculate_entropy(&guess, &candidate_refs);

        // Two patterns, equal probability = 1 bit
        assert!((entropy - 1.0).abs() < 0.001);
    }

    #[test]
    fn calculate_entropy_empty_candidates() {
        let guess = Word::new("crane").unwrap();
        let candidates: Vec<&Word> = vec![];

        let entropy = calculate_entropy(&guess, &candidates);
        assert!((entropy - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn group_by_pattern_works() {
        let guess = Word::new("crane").unwrap();
        let candidates = [
            Word::new("slate").unwrap(), // Different pattern than crane
            Word::new("crate").unwrap(), // Different pattern than slate
        ];
        let candidate_refs: Vec<&Word> = candidates.iter().collect();

        let groups = group_by_pattern(guess, &candidate_refs);

        // Should have 2 different patterns with 1 candidate each
        let non_zero_count = groups.iter().filter(|&&c| c > 0).count();
        assert_eq!(non_zero_count, 2);
        assert_eq!(groups.iter().sum::<usize>(), 2);
    }

    #[test]
    fn calculate_metrics_perfect_binary_split() {
        // Two candidates with perfect 50/50 split
        let guess = Word::new("slate").unwrap();
        let candidates = [
            Word::new("slate").unwrap(), // Perfect match
            Word::new("zzzzz").unwrap(), // No match
        ];
        let candidate_refs: Vec<&Word> = candidates.iter().collect();

        let metrics = calculate_metrics(&guess, &candidate_refs);

        // Entropy: H = -(0.5*log2(0.5) + 0.5*log2(0.5)) = 1.0
        assert!((metrics.entropy - 1.0).abs() < 0.001);

        // Expected remaining: 0.5*1 + 0.5*1 = 1.0
        assert!((metrics.expected_remaining - 1.0).abs() < 0.001);

        // Max partition: max(1, 1) = 1
        assert_eq!(metrics.max_partition, 1);
    }

    #[test]
    fn calculate_metrics_skewed_distribution() {
        // 4 candidates: 1 match + 3 non-matches
        // If guess produces pattern where 3 candidates have same pattern, 1 different
        let guess = Word::new("abcde").unwrap();
        let candidates = [
            Word::new("fghij").unwrap(), // All gray
            Word::new("fghik").unwrap(), // All gray
            Word::new("fghil").unwrap(), // All gray
            Word::new("abcde").unwrap(), // Perfect match
        ];
        let candidate_refs: Vec<&Word> = candidates.iter().collect();

        let metrics = calculate_metrics(&guess, &candidate_refs);

        // Two patterns: 3 candidates (p=0.75), 1 candidate (p=0.25)
        // Entropy: H = -(0.75*log2(0.75) + 0.25*log2(0.25))
        //          = -(0.75*-0.415 + 0.25*-2.0)
        //          = -(-0.311 - 0.5) = 0.811
        assert!((metrics.entropy - 0.811).abs() < 0.01);

        // Expected remaining: 0.75*3 + 0.25*1 = 2.25 + 0.25 = 2.5
        assert!((metrics.expected_remaining - 2.5).abs() < 0.001);

        // Max partition: max(3, 1) = 3
        assert_eq!(metrics.max_partition, 3);
    }

    #[test]
    fn calculate_metrics_empty_candidates() {
        let guess = Word::new("crane").unwrap();
        let candidates: Vec<&Word> = vec![];

        let metrics = calculate_metrics(&guess, &candidates);

        // Empty candidates → zero metrics
        assert!((metrics.entropy - 0.0).abs() < f64::EPSILON);
        assert!((metrics.expected_remaining - 0.0).abs() < f64::EPSILON);
        assert_eq!(metrics.max_partition, 0);
    }

    #[test]
    fn calculate_metrics_single_candidate() {
        let guess = Word::new("crane").unwrap();
        let candidates = [Word::new("slate").unwrap()];
        let candidate_refs: Vec<&Word> = candidates.iter().collect();

        let metrics = calculate_metrics(&guess, &candidate_refs);

        // Single candidate → deterministic outcome
        // Entropy: H = -(1.0*log2(1.0)) = 0
        assert!((metrics.entropy - 0.0).abs() < 0.001);

        // Expected remaining: 1.0*1 = 1.0
        assert!((metrics.expected_remaining - 1.0).abs() < 0.001);

        // Max partition: 1
        assert_eq!(metrics.max_partition, 1);
    }
}
