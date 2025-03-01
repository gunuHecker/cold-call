from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Deepgram Speech-to-Text API is running!"}

@app.post("/api/v1/processAudio")
async def process_audio(name: str = Form(...), purpose: str = Form(...), audio: UploadFile = File(...)):
    # Print received data for debugging
    print(f"Received name: {name}")
    print(f"Received purpose: {purpose}")
    print(f"Received audio file: {audio.filename}, Content Type: {audio.content_type}")

    # Read the audio file (for checking if it's being received correctly)
    audio_bytes = await audio.read()
    print(f"Audio file size: {len(audio_bytes)} bytes")

    return {"message": "Audio received successfully", "file_size": len(audio_bytes)}