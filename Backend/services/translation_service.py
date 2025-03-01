from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load Hinglish → English model
tokenizer_hi_en = AutoTokenizer.from_pretrained("rudrashah/RLM-hinglish-translator")
model_hi_en = AutoModelForCausalLM.from_pretrained("rudrashah/RLM-hinglish-translator")

def hinglish_to_english(text):
    template = "Hinglish:\n{hi_en}\n\nEnglish:\n{en}"
    input_text = tokenizer_hi_en(template.format(hi_en=text, en=""), return_tensors="pt")
    output = model_hi_en.generate(**input_text)
    return tokenizer_hi_en.decode(output[0])

# Load English → Hinglish model
from unsloth import FastLanguageModel

tokenizer_en_hi = None
model_en_hi = None

def load_hinglish_model():
    global tokenizer_en_hi, model_en_hi
    model_en_hi, tokenizer_en_hi = FastLanguageModel.from_pretrained(
        "Hinglish-Project/llama-3-8b-English-to-Hinglish",
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True
    )

def english_to_hinglish(text):
    if model_en_hi is None or tokenizer_en_hi is None:
        load_hinglish_model()
    
    prompt = f"### Instrucion: Translate given text to Hinglish Text:\n\n### Input:\n{text}\n\n### Response:\n"
    inputs = tokenizer_en_hi([prompt], return_tensors="pt").to("cuda")
    outputs = model_en_hi.generate(**inputs, max_new_tokens=2048)
    return tokenizer_en_hi.batch_decode(outputs)[0].split("### Response:\n")[1].split("<|end_of_text|>")[0]
