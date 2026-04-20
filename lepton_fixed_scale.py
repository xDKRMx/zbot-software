import subprocess, numpy as np, cv2

DEV = "/dev/video2"
W, H = 80, 60

# Fixed display range (raw counts). Tune these once, then keep constant.
# Formula: T = raw/100 -273.15
MINV = 28815   # ~15°C
MAXV = 33315   # ~90°C was 36315

cmd = [
    "ffmpeg", "-loglevel", "quiet",
    "-f", "v4l2",
    "-input_format", "gray16le",
    "-video_size", f"{W}x{H}",
    "-i", DEV,
    "-f", "rawvideo", "-pix_fmt", "gray16le", "-"
]
p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
frame_bytes = W * H * 2

while True:
    buf = p.stdout.read(frame_bytes)
    if len(buf) < frame_bytes:
        break
    img = np.frombuffer(buf, dtype=np.uint16).reshape(H, W)

    x = np.clip(img, MINV, MAXV)
    x = ((x - MINV) * (255.0 / (MAXV - MINV))).astype(np.uint8)

    x = cv2.resize(x, (W*8, H*8), interpolation=cv2.INTER_NEAREST)
    color = cv2.applyColorMap(x, cv2.COLORMAP_JET)

    cv2.imshow("Lepton fixed scale (ESC to quit)", color)
    if cv2.waitKey(1) & 0xFF == 27:
        break

p.terminate()
cv2.destroyAllWindows()
