from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import websockets
import asyncio
import openai
import pyttsx3
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
import os

app = FastAPI()

# Load API keys from environment variables
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Deepgram Client
deepgram = DeepgramClient(api_key=DEEPGRAM_API_KEY)


# WebSocket connections
connections = {}

# Implement Speech-to-Text (STT) with Deepgram
async def transcribe_audio(websocket: WebSocket, session_id: str):
    """
    Handles real-time speech-to-text transcription using Deepgram.
    """
    dg_connection = deepgram.listen.websocket.v("1")

    async def on_message(self, result, **kwargs):
        sentence = result.channel.alternatives[0].transcript
        if sentence:
            print(f"User: {sentence}")
            await websocket.send_text(sentence)  # Send transcribed text to frontend
            response = await generate_response(sentence)  # Get AI-generated response
            await websocket.send_text(response)
            await text_to_speech(response)  # Convert response to speech

    dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

    options = LiveOptions(model="nova-3")

    if not dg_connection.start(options):
        await websocket.send_text("Failed to start transcription session.")
        return

    connections[session_id] = websocket

    try:
        while True:
            data = await websocket.receive_bytes()
            dg_connection.send(data)

    except WebSocketDisconnect:
        print(f"Client {session_id} disconnected.")
    finally:
        dg_connection.finish()
        del connections[session_id]


# Implement LLM for AI Responses (GPT/OpenAI)
async def generate_response(user_input: str) -> str:
    """
    Uses OpenAI API to generate AI-based responses in Hinglish.
    """
    openai.api_key = OPENAI_API_KEY

    prompt = f"""
    You are an AI agent conducting a cold call in Hinglish. Keep it natural and engaging.
    User: {user_input}
    AI:
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
    )

    return response["choices"][0]["message"]["content"]

# Implement Text-to-Speech (TTS)

async def text_to_speech(text: str):
    """
    Converts AI-generated text to speech and saves it as an audio file.
    """
    engine = pyttsx3.init()
    engine.save_to_file(text, "response.mp3")
    engine.runAndWait()

# Implement WebSocket for Real-time Audio Processing
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time speech-to-text.
    """
    await websocket.accept()
    session_id = str(id(websocket))  # Unique session ID
    print(f"New connection: {session_id}")

    await transcribe_audio(websocket, session_id)

@app.get("/")
async def root():
    return {"message": "Deepgram Speech-to-Text API is running!"}

# Test the WebSocket API
async def test_ws():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        await websocket.send("Hello AI, schedule a demo!")
        response = await websocket.recv()
        print(f"AI Response: {response}")

asyncio.run(test_ws())