# Speaker Sensor Integration Guide

## 🔊 Hardware: Mini Speaker Module

Based on the image provided, you're using a **mini speaker module** (likely 8Ω 0.5W or similar) with a 2-pin JST connector.

### Hardware Specs (Typical)
- **Type**: Passive speaker (requires amplifier)
- **Impedance**: 8Ω
- **Power**: 0.5W - 1W
- **Connector**: 2-pin JST (red = +, black = GND)
- **Diameter**: ~28-36mm

## 🎯 Integration Strategy

### Option 1: Direct Raspberry Pi Audio (Recommended for Testing)
**Pros**: Simple, no extra hardware
**Cons**: Low volume, requires audio jack or GPIO PWM

```python
# Using pygame for audio playback
import pygame
from gtts import gTTS
import os

def speak_response(text: str):
    """Convert text to speech and play through speaker."""
    # Generate TTS audio
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save("temp_response.mp3")
    
    # Play through speaker
    pygame.mixer.init()
    pygame.mixer.music.load("temp_response.mp3")
    pygame.mixer.music.play()
    
    # Wait for playback to finish
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    
    # Cleanup
    os.remove("temp_response.mp3")
```

**Wiring**:
- Speaker red → Raspberry Pi audio jack (left/right channel)
- Speaker black → Raspberry Pi audio jack (ground)
- **Note**: You may need a small amplifier for better volume

### Option 2: I2S DAC + Amplifier (Recommended for Production)
**Pros**: Better audio quality, louder volume
**Cons**: Requires additional hardware

**Hardware needed**:
- MAX98357A I2S amplifier board (~$5)
- Connects to Raspberry Pi GPIO pins

**Wiring**:
```
Raspberry Pi GPIO → MAX98357A → Speaker
- GPIO 18 (BCK)   → BCLK
- GPIO 19 (LRCK)  → LRC
- GPIO 21 (DIN)   → DIN
- 5V              → VIN
- GND             → GND
                  → Speaker terminals
```

### Option 3: USB Audio Adapter (Easiest)
**Pros**: Plug-and-play, good quality
**Cons**: Uses USB port

Simply plug a USB audio adapter into Raspberry Pi and connect speaker to 3.5mm jack.

## 🤖 Software Integration

### Method 1: Read GLMCurrentResponse.txt and Speak (Recommended)

```python
import time
import os
from gtts import gTTS
import pygame

# Initialize pygame mixer
pygame.mixer.init()

last_response = ""
response_file = "GLMCurrentResponse.txt"

print("[SPEAKER] Monitoring GLMCurrentResponse.txt...")

while True:
    try:
        if os.path.exists(response_file):
            with open(response_file, "r", encoding="utf-8") as f:
                current_response = f.read().strip()
            
            # Only speak if response has changed
            if current_response and current_response != last_response:
                print(f"[SPEAKER] New response detected: {current_response[:50]}...")
                
                # Generate and play TTS
                tts = gTTS(text=current_response, lang='en', slow=False)
                tts.save("temp_response.mp3")
                
                pygame.mixer.music.load("temp_response.mp3")
                pygame.mixer.music.play()
                
                # Wait for playback
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                
                os.remove("temp_response.mp3")
                last_response = current_response
                print("[SPEAKER] Playback complete.")
        
        else:
            print("[SPEAKER] Waiting for GLMCurrentResponse.txt...")
    
    except Exception as e:
        print(f"[SPEAKER] Error: {e}")
    
    time.sleep(2)  # Check every 2 seconds
```

### Method 2: Direct Integration with Orchestrator

Update `orchestrator.py` to call TTS directly:

```python
def _trigger_audio_output(self, markdown: str) -> None:
    """Trigger audio output via speaker."""
    try:
        from gtts import gTTS
        import pygame
        import tempfile
        
        # Generate TTS
        tts = gTTS(text=markdown, lang='en', slow=False)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
            tts.save(tmp.name)
            
            # Play
            pygame.mixer.init()
            pygame.mixer.music.load(tmp.name)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            os.unlink(tmp.name)
        
        print(f"[ORCHESTRATOR] Audio output complete.")
    
    except Exception as exc:
        print(f"[ORCHESTRATOR] Audio output failed: {exc}")
```

## 📦 Required Packages

```bash
pip install gTTS pygame
```

For Raspberry Pi, also install system audio dependencies:
```bash
sudo apt-get update
sudo apt-get install -y python3-pygame libsdl2-mixer-2.0-0
```

## 🎛️ Volume Control

### Software Volume (Raspberry Pi)
```bash
# Set volume to 80%
amixer set PCM 80%

# Or in Python:
import os
os.system("amixer set PCM 80%")
```

### Hardware Volume
If using an amplifier board (MAX98357A), add a potentiometer between speaker and amp for manual volume control.

## 🚀 Complete Raspberry Pi Setup

### 1. Install Dependencies
```bash
sudo apt-get update
sudo apt-get install -y python3-pygame libsdl2-mixer-2.0-0
pip install gTTS pygame python-dotenv
```

### 2. Create Speaker Service Script
Save as `speaker_service.py`:

```python
#!/usr/bin/env python3
import time
import os
from gtts import gTTS
import pygame

pygame.mixer.init()
last_response = ""
response_file = "/home/pi/zbot-eyes/GLMCurrentResponse.txt"

print("[SPEAKER] Z-BOT Speaker Service Started")

while True:
    try:
        if os.path.exists(response_file):
            with open(response_file, "r", encoding="utf-8") as f:
                current_response = f.read().strip()
            
            if current_response and current_response != last_response:
                print(f"[SPEAKER] Speaking: {current_response[:50]}...")
                
                tts = gTTS(text=current_response, lang='en', slow=False)
                tts.save("/tmp/zbot_response.mp3")
                
                pygame.mixer.music.load("/tmp/zbot_response.mp3")
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                
                os.remove("/tmp/zbot_response.mp3")
                last_response = current_response
    
    except Exception as e:
        print(f"[SPEAKER] Error: {e}")
    
    time.sleep(2)
```

### 3. Make it Executable
```bash
chmod +x speaker_service.py
```

### 4. Run on Boot (systemd service)
Create `/etc/systemd/system/zbot-speaker.service`:

```ini
[Unit]
Description=Z-BOT Speaker Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/zbot-eyes
ExecStart=/usr/bin/python3 /home/pi/zbot-eyes/speaker_service.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable zbot-speaker.service
sudo systemctl start zbot-speaker.service
sudo systemctl status zbot-speaker.service
```

## 🎯 Expected Behavior

1. **Unified runner** detects events → sends to orchestrator
2. **Orchestrator** aggregates events → sends to GLM with logs + image
3. **GLM** returns conversational response (short, 3 sentences max)
4. **Orchestrator** writes response to `GLMCurrentResponse.txt`
5. **Speaker service** detects file change → converts to speech → plays through speaker
6. **Judges hear**: "Hey there! I've detected a safety net covering 42% of my view, but I'm also spotting some debris at 3% coverage. I recommend we pause and alert the maintenance team!"

## ✅ Testing

### Test 1: Manual TTS
```python
from gtts import gTTS
import pygame

text = "Hello judges! I am Z-BOT, your friendly wall-climbing robot!"
tts = gTTS(text=text, lang='en', slow=False)
tts.save("test.mp3")

pygame.mixer.init()
pygame.mixer.music.load("test.mp3")
pygame.mixer.music.play()

while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)
```

### Test 2: File Monitoring
```bash
# Terminal 1: Run speaker service
python speaker_service.py

# Terminal 2: Simulate GLM response
echo "Hey there! I've detected a safety net ahead!" > GLMCurrentResponse.txt
```

You should hear the speaker play the response!

## 🔧 Troubleshooting

### No Sound
1. Check speaker wiring (red to +, black to GND)
2. Verify Raspberry Pi audio output: `aplay -l`
3. Test with: `speaker-test -t wav -c 2`
4. Check volume: `amixer get PCM`

### Low Volume
- Use amplifier board (MAX98357A)
- Or use USB audio adapter with powered speakers

### Distorted Audio
- Reduce volume: `amixer set PCM 60%`
- Check speaker impedance (should be 8Ω)

## 🎉 Summary

Your speaker will work perfectly with this system! The flow is:

```
Detection → GLM (logs + image) → Response → GLMCurrentResponse.txt → Speaker → Judges hear it!
```

**Recommended setup for Challenge Cup**:
- Use **Method 1** (file monitoring) for reliability
- Add **MAX98357A amplifier** for better volume
- Run **speaker service** as systemd service for auto-start
- Test thoroughly before demo!
