# Wordle Solver

![CI](https://github.com/nsheely/Wordle_Solver/actions/workflows/build.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Rust](https://img.shields.io/badge/rust-2024-orange.svg)

A Wordle solver in Rust using information theory and game theory to achieve near optimal performance.

## What It Does

Solves Wordle puzzles by combining multiple strategies:
- **Entropy maximization** - maximizes information gain from each guess
- **Minimax** - minimizes worst-case remaining candidates
- **Hybrid** - balances both approaches adaptively

**3.4334 ± 0.0004 (SE) average guesses** - 99.64% of optimal 3.421*

*Optimal value from [Bertsimas & Paskov (2022)](https://auction-upload-files.s3.amazonaws.com/Wordle_Paper_Final.pdf)

## Usage

### Build

```bash
cargo build --release
```

### Getting Help

```bash
# Main help
wordle_solver --help

# Help for specific commands
wordle_solver play --help
wordle_solver solve --help
```

### Commands

**Interactive TUI** - Full-screen interface with visualizations:
```bash
wordle_solver play
```

**Simple CLI** - Text-based interactive solver:
```bash
wordle_solver simple

# With a specific strategy
wordle_solver simple --strategy minimax
```

**Solve a specific word** - See how the solver would solve it:
```bash
wordle_solver solve CRANE

# With verbose output
wordle_solver solve CRANE --verbose
```

**Analyze a word** - See its entropy and information value:
```bash
wordle_solver analyze SALET
```

**Benchmark** - Test performance on random sample:
```bash
wordle_solver benchmark --count 100
```

**Test all answers** - Full evaluation on all 2,315 words:
```bash
wordle_solver test-all
```

## Strategies

Use `--strategy` or `-s` to select:
- `adaptive` (default) - 5-tier strategy that adapts based on remaining candidates
- `entropy` - Pure information maximization
- `minimax` - Pure worst-case minimization
- `hybrid` - Weighted combination
- `random` - Random selection (baseline)

Example:
```bash
wordle_solver simple --strategy minimax
```

## Performance

Measured over 150 independent test runs, each solving all 2,315 Wordle answer words (347,250 total solves).

- **Average guesses**: 3.4334 ± 0.0004 (SE) - 99.64% of optimal 3.421
- **Standard deviation**: 0.0050 guesses
- **95% confidence interval**: 3.433 - 3.434 guesses
- **Observed range**: 3.419 - 3.445 guesses (99.30% - 100.06% optimal*)
- **Success rate**: 100% within 6 guesses

* Optimal reference used a 10,657-word guess pool. This solver uses a 12,972-word guess pool, which explains occasional runs performing at 100%+ (different problem spaces).


**Typical distribution:**
- 2 guesses: 78-79 words (3.4%)
- 3 guesses: 1,203-1,204 words (52%)
- 4 guesses: 980-984 words (42%)
- 5 guesses: 47-50 words (2%)
- 6 guesses: 1-3 words (0.1%)

## How the Adaptive Strategy Works

Uses different tactics based on how many candidates remain:

1. **81+ candidates**: Pure entropy - maximize information gain
2. **22-80 candidates**: Entropy with minimax tiebreaker
3. **16-21 candidates**: Hybrid scoring (entropy × 100) - (max_partition × 10)
4. **3-15 candidates**: Minimax-first with 20% candidate preference
5. **1-2 candidates**: Random selection

The strategy automatically switches tactics as candidates are eliminated.

## Project Structure

```
src/
├── main.rs              # CLI entry point
├── lib.rs               # Library root
├── commands/            # Command implementations
├── core/                # Core types (Word, Pattern)
├── solver/              # Solving strategies
│   ├── adaptive.rs      # 5-tier adaptive strategy
│   ├── entropy/         # Entropy calculations
│   ├── minimax/         # Minimax selection
│   └── selection/       # Hybrid selection logic
├── interactive/         # TUI mode
├── output/              # Display formatting
└── wordlists/           # Word list management
```

## License

MIT
