# vhs script for the README demo. Regenerate from project root with:
#   vhs assets/demo.tape
#
# Walks the solver through three target words back-to-back. Each path is
# the deterministic sequence the adaptive strategy produces against the
# embedded wordlist - if the solver changes, re-derive with `cargo run --
# solve <WORD>`.
#
#   1. TRACE  - 3 guesses
#        SALET  -> - Y - Y Y
#        GRATE  -> - G G Y G
#        TRACE  -> G G G G G
#
#   2. QUIRK  - 4 guesses (cold start: SALET is all gray)
#        SALET  -> - - - - -
#        CORNU  -> - - Y - Y
#        RUGBY  -> Y G - - -
#        QUIRK  -> G G G G G
#
#   3. FLAME  - 4 guesses
#        SALET  -> - Y Y Y -
#        GLARE  -> - G G - G
#        POMBE  -> - - Y - G
#        FLAME  -> G G G G G

Output assets/demo.gif

Set Shell bash
Set FontSize 18
Set Width 1400
Set Height 920
Set Theme "Monokai Vivid"
Set LineHeight 1.4
Set Padding 24
Set TypingSpeed 60ms
Set Framerate 50

Type "./target/release/wordle_solver play"
Enter
Sleep 1500ms

# --- TRACE ---
Type "-Y-YY"
Sleep 400ms
Enter
Sleep 1200ms

Type "-GGYG"
Sleep 400ms
Enter
Sleep 1200ms

Type "GGGGG"
Sleep 400ms
Enter
Sleep 1800ms

# New game
Type "n"
Sleep 1000ms

# --- QUIRK ---
Type "-----"
Sleep 400ms
Enter
Sleep 1200ms

Type "--Y-Y"
Sleep 400ms
Enter
Sleep 1200ms

Type "YG---"
Sleep 400ms
Enter
Sleep 1200ms

Type "GGGGG"
Sleep 400ms
Enter
Sleep 1800ms

# New game
Type "n"
Sleep 1000ms

# --- FLAME ---
Type "-YYY-"
Sleep 400ms
Enter
Sleep 1200ms

Type "-GG-G"
Sleep 400ms
Enter
Sleep 1200ms

Type "--Y-G"
Sleep 400ms
Enter
Sleep 1200ms

Type "GGGGG"
Sleep 400ms
Enter
Sleep 2500ms

# Quit
Type "q"
Sleep 500ms
