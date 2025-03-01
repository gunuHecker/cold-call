import os
import aiohttp
from app.utils.config import DEEPGRAM_API_KEY

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"

async def transcribe_audio(audio_bytes):
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "audio/wav"}
    params = {"punctuate": "true", "language": "hi"}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(DEEPGRAM_URL, headers=headers, params=params, data=audio_bytes) as response:
                deepgram_response = await response.json()
        return deepgram_response.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")
    except Exception as e:
        print(f"Deepgram Error: {e}")
        return None