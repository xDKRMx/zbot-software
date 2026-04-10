#!/usr/bin/env python3
"""List young, cute female voices from Edge TTS."""

import asyncio
import edge_tts

async def list_voices():
    voices = await edge_tts.list_voices()
    
    # Filter for young, cute female voices
    young_cute = []
    
    for voice in voices:
        name = voice["ShortName"]
        gender = voice.get("Gender", "")
        locale = voice.get("Locale", "")
        
        # Look for young/child/girl voices
        if gender == "Female":
            # Check for young indicators in voice name
            if any(keyword in name.lower() for keyword in ["xiaoxiao", "xiaoyi", "yunxi", "xiaomo", "yoyo", "aria", "sara", "girl", "child", "young"]):
                young_cute.append({
                    "name": name,
                    "locale": locale,
                    "gender": gender
                })
    
    print("\n🎀 Young & Cute Female Voices 🎀")
    print("=" * 80)
    for v in young_cute:
        print(f"{v['name']:40} | {v['locale']:10} | {v['gender']}")
    print("=" * 80)
    print(f"\nTotal: {len(young_cute)} voices\n")

if __name__ == "__main__":
    asyncio.run(list_voices())
