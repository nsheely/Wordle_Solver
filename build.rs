//! Build script to generate embedded word lists
//!
//! Reads word list files and generates Rust source code with const arrays.

use std::env;
use std::fs;
use std::io::Write;
use std::path::Path;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    // Generate answers list
    generate_word_list(
        "data/answers.txt",
        &Path::new(&out_dir).join("answers.rs"),
        "ANSWERS",
        "Official Wordle answer words",
    );

    // Generate allowed list (complete set)
    generate_word_list(
        "data/allowed_complete.txt",
        &Path::new(&out_dir).join("allowed.rs"),
        "ALLOWED",
        "All allowed guessable words",
    );

    // Rebuild if word lists change
    println!("cargo:rerun-if-changed=data/answers.txt");
    println!("cargo:rerun-if-changed=data/allowed_complete.txt");
}

fn generate_word_list(input_path: &str, output_path: &Path, const_name: &str, doc_comment: &str) {
    let content = fs::read_to_string(input_path)
        .unwrap_or_else(|e| panic!("Failed to read {input_path}: {e}"));

    let words: Vec<&str> = content.lines().collect();
    let count = words.len();

    let mut output = fs::File::create(output_path)
        .unwrap_or_else(|e| panic!("Failed to create {}: {e}", output_path.display()));

    writeln!(output, "// Generated word list").unwrap();
    writeln!(output, "//").unwrap();
    writeln!(output, "// {doc_comment}").unwrap();
    writeln!(output).unwrap();
    writeln!(output, "/// {doc_comment}").unwrap();
    writeln!(output, "pub const {const_name}: &[&str] = &[").unwrap();

    for word in words {
        writeln!(output, "    \"{}\",", word.trim()).unwrap();
    }

    writeln!(output, "];").unwrap();
    writeln!(output).unwrap();
    writeln!(output, "/// Number of words in {const_name}").unwrap();
    writeln!(output, "pub const {const_name}_COUNT: usize = {count};").unwrap();
}
