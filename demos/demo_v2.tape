# Vectro 2.0 Overdrive — terminal demo recording
# Produces: demos/demo_v2.gif
# Run with: vhs demos/demo_v2.tape

Output demos/demo_v2.gif

Set Theme "Catppuccin Mocha"
Set FontSize 13
Set Width 900
Set Height 620
Set Padding 18
Set Framerate 30
Set PlaybackSpeed 1.5

# Start with a clean prompt
Type "cd /Users/wscholl/vectro"
Enter
Sleep 300ms

Type "python3 demos/demo_v2.py --quick"
Enter
Sleep 500ms

# Section 1 — Core Compression
Sleep 2s

# Section 2 — Streaming
Sleep 1.5s

# Section 3 — INT2 / Adaptive
Sleep 1s

# Section 4 — Vector DB
Sleep 1s

# Section 5 — Arrow
Sleep 1.5s

# Section 6 — Migration
Sleep 1.5s

# Section 7 — Benchmark table
Sleep 2s

# Section 8 — CLI
Sleep 1s

# Final scorecard
Sleep 2.5s
