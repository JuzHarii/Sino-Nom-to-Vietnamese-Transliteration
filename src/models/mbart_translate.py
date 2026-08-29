import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "Ganth1811/sino-nom-to-vietnamese-mbart-5epoch"
tokenizer = None
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_mbart_model():
    global tokenizer, model
    if model is None or tokenizer is None:
        print(f"Loading mbart model: {MODEL_NAME} on {device}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        model.to(device)
        print("mbart model loaded.")
def mbart_translate(text):
    if model is None or tokenizer is None:
        load_mbart_model()

    tokenizer.src_lang = "zh_CN"    
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
    
    dynamic_max_length = inputs.input_ids.shape[1] * 2 + 10
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["vi_VN"],
            max_new_tokens=dynamic_max_length, 
            num_beams=4, 
            early_stopping=True, 
            repetition_penalty=2.5,     
            no_repeat_ngram_size=2      
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
