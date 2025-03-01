from transformers import pipeline

# Load Transformer model
qa_pipeline = pipeline("text-generation", model="Qwen/Qwen2.5-Coder-32B-Instruct")

def generate_ai_response(text):
    response = qa_pipeline(text, max_length=100)
    return response[0]["generated_text"]
