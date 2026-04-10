#!/usr/bin/env python3
"""Anime-style voice TTS speaker for Z-BOT GLM responses.

This script monitors GLMCurrentResponse.txt and speaks new responses
using Microsoft Edge TTS with a cute anime girl voice.
"""

import asyncio
import os
import time
from pathlib import Path

try:
    import edge_tts
    import pygame
except ImportError:
    print("ERROR: Missing dependencies. Run: pip install edge-tts pygame")
    exit(1)


# Anime girl voice options (Microsoft Edge TTS)
# Focus on YOUNG, CUTE voices (not mature ladies)
ANIME_VOICES = {
    "japanese_cute": "ja-JP-NanamiNeural",      # Cute Japanese girl (heavy accent)
    "japanese_cheerful": "ja-JP-AoiNeural",     # Cheerful Japanese girl (heavy accent)
    "chinese_cute": "zh-CN-XiaoxiaoNeural",     # YOUNG Chinese girl 
    "chinese_warm": "zh-CN-XiaoyiNeural",       # Warm Chinese girl
    "korean_cute": "ko-KR-SunHiNeural",         # Cute Korean girl
    "english_young": "en-US-AriaNeural",        # YOUNG American girl 
    "indian_male": "en-IN-PrabhatNeural",        # Indian male English
    "vietnamese_female": "vi-VN-HoaiMyNeural",   # Vietnamese female
    "vietnamese_male": "vi-VN-NamMinhNeural",    # Vietnamese male
    "thai_female": "th-TH-PremwadeeNeural",      # Thai female
    "thai_male": "th-TH-NiwatNeural",            # Thai male
    "indonesian_female": "id-ID-GadisNeural",     # Indonesian female
    "indonesian_male": "id-ID-ArdiNeural",        # Indonesian male
    "english_mature": "en-US-JennyNeural",      # Mature English (not young)
    "english_british": "en-GB-SoniaNeural",     # British (not young)
}

# Default voice (Xiaoxiao - YOUNG Chinese girl, for Chinese-only responses)
DEFAULT_VOICE = ANIME_VOICES["chinese_cute"]

# You can change this to any voice from ANIME_VOICES
SELECTED_VOICE = DEFAULT_VOICE


async def text_to_speech(text: str, output_file: str, voice: str = DEFAULT_VOICE):
    """Convert text to speech using Edge TTS with anime voice."""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)


def play_audio(audio_file: str):
    """Play audio file using pygame."""
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    
    # Wait for playback to finish
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    
    pygame.mixer.music.unload()


def speak_text(text: str, voice: str = DEFAULT_VOICE, bilingual: bool = True):
    """Speak text with anime voice. If bilingual, speaks both English and Chinese."""
    temp_file = "temp_anime_voice.mp3"
    
    try:
        # Check if response contains bilingual separator
        if bilingual and "---" in text:
            parts = text.split("---")
            english_text = parts[0].strip()
            chinese_text = parts[1].strip() if len(parts) > 1 else ""
            
            if english_text:
                print(f"[ANIME VOICE] 🇺🇸 English: {english_text[:50]}...")
                asyncio.run(text_to_speech(english_text, temp_file, voice))
                play_audio(temp_file)
                time.sleep(0.5)  # Brief pause between languages
            
            if chinese_text:
                print(f"[ANIME VOICE] 🇨🇳 Chinese: {chinese_text[:50]}...")
                # Use Chinese voice for Chinese text
                chinese_voice = "zh-CN-XiaoxiaoNeural"
                asyncio.run(text_to_speech(chinese_text, temp_file, chinese_voice))
                play_audio(temp_file)
            
            print("[ANIME VOICE] Bilingual playback complete! ✨")
        
        else:
            # Single language mode
            print(f"[ANIME VOICE] Generating speech with {voice}...")
            asyncio.run(text_to_speech(text, temp_file, voice))
            print(f"[ANIME VOICE] Playing: {text[:50]}...")
            play_audio(temp_file)
            print("[ANIME VOICE] Playback complete! ✨")
    
    except Exception as exc:
        print(f"[ANIME VOICE] Error: {exc}")
    
    finally:
        # Cleanup
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass


def monitor_glm_response(
    response_file: str = "GLMCurrentResponse.txt",
    voice: str = DEFAULT_VOICE,
    check_interval: float = 2.0,
    bilingual: bool = True,
):
    """Monitor GLMCurrentResponse.txt and speak new responses with anime voice."""
    last_response = ""
    
    print("=" * 60)
    print("🎀 Z-BOT Anime Voice Speaker Started! 🎀")
    print("=" * 60)
    print(f"Voice (English): {voice}")
    print(f"Voice (Chinese): zh-CN-XiaoxiaoNeural")
    print(f"Monitoring: {response_file}")
    print(f"Bilingual mode: {'ON 🇺🇸🇨🇳' if bilingual else 'OFF'}")
    print(f"Check interval: {check_interval}s")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    while True:
        try:
            if os.path.exists(response_file):
                with open(response_file, "r", encoding="utf-8") as f:
                    current_response = f.read().strip()
                
                # Only speak if response has changed
                if current_response and current_response != last_response:
                    print(f"\n[NEW RESPONSE] {current_response[:100]}...\n")
                    speak_text(current_response, voice, bilingual=bilingual)
                    last_response = current_response
            
            else:
                print(f"[WAITING] {response_file} not found yet...")
            
            time.sleep(check_interval)
        
        except KeyboardInterrupt:
            print("\n[ANIME VOICE] Stopping... Bye bye! 👋✨")
            break
        
        except Exception as exc:
            print(f"[ANIME VOICE] Error: {exc}")
            time.sleep(check_interval)


def list_available_voices():
    """List all available anime-style voices."""
    print("\n🎀 Available Anime Voices 🎀")
    print("=" * 60)
    for name, voice_id in ANIME_VOICES.items():
        print(f"  {name:20} → {voice_id}")
    print("=" * 60)
    print(f"\nCurrent voice: {SELECTED_VOICE}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Anime-style voice TTS speaker for Z-BOT"
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="english_young",
        choices=list(ANIME_VOICES.keys()),
        help="Select anime voice style (default: english_young - Aria)"
    )
    parser.add_argument(
        "--file",
        type=str,
        default="GLMCurrentResponse.txt",
        help="GLM response file to monitor"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Check interval in seconds"
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List all available voices and exit"
    )
    parser.add_argument(
        "--test",
        type=str,
        help="Test voice with custom text"
    )
    parser.add_argument(
        "--bilingual",
        action="store_true",
        default=False,
        help="Enable bilingual mode (English + Chinese)"
    )
    parser.add_argument(
        "--no-bilingual",
        action="store_false",
        dest="bilingual",
        help="Disable bilingual mode (English only)"
    )
    
    args = parser.parse_args()
    
    if args.list_voices:
        list_available_voices()
        exit(0)
    
    selected_voice = ANIME_VOICES[args.voice]
    
    if args.test:
        print(f"[TEST MODE] Testing voice: {selected_voice}")
        speak_text(args.test, selected_voice, bilingual=False)
        exit(0)
    
    # Start monitoring
    monitor_glm_response(
        response_file=args.file,
        voice=selected_voice,
        check_interval=args.interval,
        bilingual=args.bilingual,
    )
