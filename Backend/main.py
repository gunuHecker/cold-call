# from fastapi import FastAPI, File, UploadFile, Form
# from fastapi.middleware.cors import CORSMiddleware
# from deepgram import Deepgram
# from dotenv import load_dotenv
# # from unsloth import FastLanguageModel
# # from session_manager import SessionManager
# import os
# import aiohttp
# import asyncio
# import json
# import uuid

# # Load environment variables from .env file
# load_dotenv()

# # Replace with your Deepgram API Key
# DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# if not DEEPGRAM_API_KEY:
#     raise ValueError("Missing API keys! Check .env file.")

# app = FastAPI()

# origins = [
#     "http://localhost.tiangolo.com",
#     "https://localhost.tiangolo.com",
#     "http://localhost",
#     "http://localhost:8080",
#     "http://localhost:5173"
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.get("/")
# async def root():
#     return {"message": "Deepgram Speech-to-Text API is running!"}

# # global session_manager
# # session_manager = SessionManager().get_instance()

# # @app.get("/api/v1/getSessionId")
# # async def get_session_id():
# #     session_id_uuid = uuid.uuid4()
# #     session_id = str(session_id_uuid)

# #     session_manager.initiate_session(session_id)
# #     return {"session_id": session_id}

# """
#     1. Use Deepgram to listen to audio
#     2. Convert Hinglish to English using rudrashah/RLM-hinglish-translator
#     3. Use Transformer to generate response in English
#     4. Convert English to Hinglish using Hinglish-Project/llama-3-8b-English-to-Hinglish
#     5. Convert this text to speech
#     6. Implement Langchain
#     7. Web Sockets
#     8. Session Management
# """

# @app.post("/api/v1/processAudio/")
# async def process_audio(name: str = Form(...), purpose: str = Form(...), audio: UploadFile = File(...)):
#     print(f"Received name: {name}")
#     print(f"Received purpose: {purpose}")
#     print(f"Received audio file: {audio.filename}, Content Type: {audio.content_type}")

#     # Read the audio file
#     audio_bytes = await audio.read()
#     print(f"Audio file size: {len(audio_bytes)} bytes")

#     # session = session_manager.get_session(session_id)

#     # Deepgram API Endpoint
#     url = "https://api.deepgram.com/v1/listen"

#     headers = {
#         "Authorization": f"Token {DEEPGRAM_API_KEY}",
#         "Content-Type": "audio/wav"
#     }

#     params = {
#         "punctuate": "true",
#         "language": "hi"  # Use Hindi
#     }

#     # Send audio to Deepgram
#     try:
#         async with aiohttp.ClientSession() as session:
#             async with session.post(url, headers=headers, params=params, data=audio_bytes) as response:
#                 deepgram_response = await response.json()

#         # 🔹 Log the full Deepgram response
#         print("Full Deepgram API Response:", deepgram_response)

#         # deepgram_json = json.dumps(deepgram_response, indent=4)
#         # print(deepgram_json)
    
#         # 🔹 Check if transcription is available
#         if "results" in deepgram_response:
#             transcript = deepgram_response["results"]["channels"][0]["alternatives"][0]["transcript"]
#             print(f"Transcribed Text (Hindi): {transcript}")

#             # # 🔹 Generate AI response using Hugging Face Model
#             # messages = [{"role": "user", "content": transcript}]
#             # completion = hf_client.chat.completions.create(
#             #     model="Qwen/Qwen2.5-Coder-32B-Instruct",
#             #     messages=messages,
#             #     max_tokens=100,
#             # )

#             # answer = completion.choices[0].message.content
#             # print("AI Response:", answer)

#             return {"message": "Audio received successfully", "transcription": transcript,
#             # "ai_response": answer,        
#             }
#         else:
#             print("Deepgram API did not return 'results'.")
#             return {"error": "Deepgram API did not return a valid response", "response": deepgram_response}

#     except Exception as e:
#         print(f"Error processing audio: {e}")
#         return {"error": "Failed to transcribe audio"}

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import aiohttp
from dotenv import load_dotenv
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
# import torch
# from unsloth import FastLanguageModel

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"

# # 🔹 Load Transformer AI Model for Text Generation
# try:
#     qa_pipeline = pipeline("text-generation", model="Qwen/Qwen2.5-Coder-32B-Instruct")
# except Exception as e:
#     print(f"❌ Error loading AI model: {e}")
#     qa_pipeline = None

# def generate_ai_response(text: str) -> str:
#     """Generate AI response from Qwen2.5-Coder model."""
#     if qa_pipeline is None:
#         return "AI Model not loaded properly."
    
#     response = qa_pipeline(text, max_length=100)
#     return response[0].get("generated_text", "").strip()

# 🔹 Speech-to-Text with Deepgram
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

# # 🔹 Load Hinglish → English Translation Model
# try:
#     tokenizer_hi_en = AutoTokenizer.from_pretrained("rudrashah/RLM-hinglish-translator")
#     model_hi_en = AutoModelForCausalLM.from_pretrained("rudrashah/RLM-hinglish-translator")
# except Exception as e:
#     print(f"❌ Error loading Hinglish → English model: {e}")
#     tokenizer_hi_en, model_hi_en = None, None

# def hinglish_to_english(text):
#     """Translate Hinglish to English."""
#     if tokenizer_hi_en is None or model_hi_en is None:
#         return "Translation model not loaded."
    
#     template = "Hinglish:\n{hi_en}\n\nEnglish:\n"
#     input_text = tokenizer_hi_en(template.format(hi_en=text), return_tensors="pt")
#     output = model_hi_en.generate(**input_text)
#     return tokenizer_hi_en.decode(output[0]).strip()

# # 🔹 Lazy Load English → Hinglish Model
# tokenizer_en_hi = None
# model_en_hi = None

# def load_hinglish_model():
#     """Load English to Hinglish Translation Model."""
#     global tokenizer_en_hi, model_en_hi
#     if tokenizer_en_hi is None or model_en_hi is None:
#         model_en_hi, tokenizer_en_hi = FastLanguageModel.from_pretrained(
#             "Hinglish-Project/llama-3-8b-English-to-Hinglish",
#             max_seq_length=512,
#             dtype=None,
#             load_in_4bit=True
#         )

# def english_to_hinglish(text):
#     """Translate English to Hinglish."""
#     if model_en_hi is None or tokenizer_en_hi is None:
#         load_hinglish_model()
    
#     prompt = f"### Instruction: Translate given text to Hinglish:\n\n### Input:\n{text}\n\n### Response:\n"
#     inputs = tokenizer_en_hi([prompt], return_tensors="pt").to("cuda")
#     outputs = model_en_hi.generate(**inputs, max_new_tokens=2048)
#     return tokenizer_en_hi.batch_decode(outputs)[0].split("### Response:\n")[1].split("<|end_of_text|>")[0].strip()

# 🔹 FastAPI Setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/api/v1/processAudio/")
async def process_audio(name: str = Form(...), purpose: str = Form(...), audio: UploadFile = File(...)):
    """Process audio file: STT → Hinglish → English → AI → Hinglish."""
    audio_bytes = await audio.read()

    # 🔹 Step 1: Speech-to-Text
    transcript = await transcribe_audio(audio_bytes)
    if not transcript:
        return {"error": "Failed to transcribe audio"}

    # 🔹 Step 2: Hinglish → English Translation
    # english_text = hinglish_to_english(transcript)

    # # 🔹 Step 3: Generate AI Response
    # ai_response = generate_ai_response(english_text)

    # if "not loaded properly" in ai_response:
    #     return {"error": "AI model failed to load"}

    # # 🔹 Step 4: English → Hinglish Translation
    # hinglish_response = english_to_hinglish(ai_response)

    return {
        "transcription": transcript,
        # "translated_english": english_text,
        # "ai_response": ai_response,
        # "final_hinglish": hinglish_response
    }