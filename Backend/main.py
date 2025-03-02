from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import aiohttp
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
# import torch
# from unsloth import FastLanguageModel

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"

# 🔹 Load Transformer AI Model for Text Generation
import torch

try:
    if torch.cuda.is_available():
        device = 0  # GPU
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = -1  # CPU
        print("⚠️ GPU not available. Falling back to CPU.")

    qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small", device=device)
except Exception as e:
    print(f"❌ Error loading AI model: {e}")
    qa_pipeline = None

def generate_ai_response(text: str) -> str:
    """Generate AI response from Flan-T5 small model."""
    if qa_pipeline is None:
        return "AI Model not loaded properly."
    
    response = qa_pipeline(text, max_length=100)
    return response[0].get("generated_text", "").strip()

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

# 🔹 Load Hinglish → English Translation Model
try:
    tokenizer_hi_en = AutoTokenizer.from_pretrained("rudrashah/RLM-hinglish-translator")
    model_hi_en = AutoModelForCausalLM.from_pretrained("rudrashah/RLM-hinglish-translator")
except Exception as e:
    print(f"❌ Error loading Hinglish → English model: {e}")
    tokenizer_hi_en, model_hi_en = None, None

def hinglish_to_english(text):
    """Translate Hinglish to English."""
    if tokenizer_hi_en is None or model_hi_en is None:
        return "Translation model not loaded."
    
    template = "Hinglish:\n{hi_en}\n\nEnglish:\n"
    input_text = tokenizer_hi_en(template.format(hi_en=text), return_tensors="pt")
    output = model_hi_en.generate(**input_text)
    return tokenizer_hi_en.decode(output[0]).strip()

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
    english_text = hinglish_to_english(transcript)

    # 🔹 Step 3: Generate AI Response
    ai_response = generate_ai_response(english_text)

    # if "not loaded properly" in ai_response:
    #     return {"error": "AI model failed to load"}

    # # 🔹 Step 4: English → Hinglish Translation
    # hinglish_response = english_to_hinglish(ai_response)

    return {
        "transcription": transcript,
        "translated_english": english_text,
        "ai_response": ai_response,
        # "final_hinglish": hinglish_response
    }