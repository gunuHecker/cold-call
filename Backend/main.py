from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import aiohttp
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM 
import torch
# from unsloth import FastLanguageModel

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"

# # ✅ Load TinyLlama model
# try:
#     model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype=torch.float16
#     ).cuda()
#     print(f"✅ Loaded {model_name} on GPU: {torch.cuda.get_device_name(0)}")
# except Exception as e:
#     print(f"❌ Error loading TinyLlama model: {e}")
#     tokenizer = None
#     model = None

# def generate_ai_response(text: str) -> str:
#     """Generate AI response using TinyLlama."""
#     if model is None or tokenizer is None:
#         return "AI Model not loaded properly."
    
#     try:
#         prompt = f"<|system|> You are a helpful assistant.\n<|user|> {text}\n<|assistant|>"
#         inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=512,
#             do_sample=True,  # Optional: for more natural responses
#             temperature=0.7,  # Optional: controls randomness
#             top_p=0.9  # Optional: nucleus sampling
#         )
#         response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return response.split("<|assistant|>")[-1].strip()
    
#     except Exception as e:
#         print(f"❌ Error during generation: {e}")
#         return "Error generating response."

# try:
#     if torch.cuda.is_available():
#         device = 0  # GPU
#         print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
#     else:
#         device = -1  # CPU
#         print("⚠️ GPU not available. Falling back to CPU.")

#     qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small", device=device)
# except Exception as e:
#     print(f"❌ Error loading AI model: {e}")
#     qa_pipeline = None

# def generate_ai_response(text: str) -> str:
#     """Generate AI response from Flan-T5 small model."""
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

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"✅ Using device: {device}")

# tokenizer_en_hi = AutoTokenizer.from_pretrained('akashgoel-id/OpenHathi-7B-English-to-Hinglish')
# model_en_hi = LlamaForCausalLM.from_pretrained(
#     'akashgoel-id/OpenHathi-7B-English-to-Hinglish',
#     torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
# ).to(device)

# def english_to_hinglish(text):
#     """Translate English to Hinglish."""
#     prompt = f"Translate from english to hinglish:\n{text}\n---\nTranslation:\n"
#     inputs = tokenizer_en_hi(prompt, return_tensors="pt").to(device)
#     generate_ids = model_en_hi.generate(inputs.input_ids, max_length=500)
#     output = tokenizer_en_hi.batch_decode(
#         generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
#     )[0]
#     # Optionally, clean out the prompt from the output if needed
#     return output.split("Translation:\n")[-1].strip()

# 🔹 FastAPI Setup
app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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

    # # 🔹 Step 3: Generate AI Response
    # ai_response = generate_ai_response(english_text)

    # if "not loaded properly" in ai_response:
    #     return {"error": "AI model failed to load"}

    # # 🔹 Step 4: English → Hinglish Translation
    # hinglish_response = english_to_hinglish(ai_response)

    return {
        "transcription": transcript,
        "translated_english": english_text,
        # "ai_response": ai_response,
        # "final_hinglish": hinglish_response
    }