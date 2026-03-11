# Vectro v3.0.0 — terminal demo recording
# Produces: demos/demo_v3.gif
# Run with: vhs demos/demo_v3.tape

Output demos/demo_v3.gif

Set Theme "Catppuccin Mocha"
Set FontSize 12
Set Width 900
Set Height 580
Set Padding 16
Set Framerate 15
Set PlaybackSpeed 1.5

# Start from project root
Type "cd /Users/wscholl/vectro"
Enter
Sleep 400ms

Type "python3 demos/gif_demo_v3.py"
Enter

# Script runs in ~46 s; give it 52 s of headroom
Sleep 52s

# Linger on the final summary screen
Sleep 2s
