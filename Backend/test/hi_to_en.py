from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("rudrashah/RLM-hinglish-translator")
model = AutoModelForCausalLM.from_pretrained("rudrashah/RLM-hinglish-translator")

template = "Hinglish:\n{hi_en}\n\nEnglish:\n{en}" #THIS IS MOST IMPORTANT, WITHOUT THIS IT WILL GIVE RANDOM OUTPUT
input_text = tokenizer(template.format(hi_en="aapka name kya hai?",en=""),return_tensors="pt")

output = model.generate(**input_text)
print(tokenizer.decode(output[0]))