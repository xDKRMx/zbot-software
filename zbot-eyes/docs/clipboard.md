# Clipboard Paste

The GUI supports pasting images from the clipboard using several fallbacks.

## Supported methods

1) **PIL ImageGrab**
   - Works on Windows/macOS, sometimes on Linux.

2) **Clipboard contains a file path**
   - Copy a local image file path and click **Paste image**.

3) **Linux (Wayland)**
   - Uses `wl-paste` if installed.
   - Install: `sudo apt install wl-clipboard`

4) **Linux (X11)**
   - Uses `xclip` or `xsel` if installed.
   - Install: `sudo apt install xclip` or `sudo apt install xsel`

5) **macOS**
   - Uses `pngpaste` if installed.
   - Install: `brew install pngpaste`

## Troubleshooting

- If you see "No image found in clipboard", try copying an image **file path**.
- On Linux, check session type: `echo $XDG_SESSION_TYPE`.
- Make sure the GUI is running in a desktop session (not headless).

