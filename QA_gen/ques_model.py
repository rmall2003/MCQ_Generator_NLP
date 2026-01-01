from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

class QuestionGenerator:
    def __init__(self, tokenizer, model, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        self.model = model
        self.model.to(self.device) 


    def _build_input(self, context, answer):
        # Enhanced prompt for better question generation
        prompt = f"Generate a clear and specific question where the answer is '{answer}'. Context: {context}"
        return prompt

    def generate_question(self, context, answer, max_length=64):
        prompt = self._build_input(context, answer)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        # Enhanced generation parameters for better quality
        out = self.model.generate(
            **inputs, 
            max_length=max_length,
            num_beams=5,  # Increased from 4 for better search
            temperature=0.7,  # Add some creativity
            do_sample=True,  # Enable sampling
            top_p=0.9,  # Nucleus sampling
            no_repeat_ngram_size=3,  # Avoid repetition
            early_stopping=True
        )
        
        q = self.tokenizer.decode(out[0], skip_special_tokens=True)
        q = q.replace("question:", "").strip()
        
        if len(q) < 6 or re.fullmatch(r'[\W_]+', q):
            return ""
        return q
