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
        prompt = f"answer: {answer} context: {context}"
        return prompt

    def generate_question(self, context, answer, max_length=64):
        prompt = self._build_input(context, answer)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        out = self.model.generate(**inputs, max_length=max_length, num_beams=4, early_stopping=True)
        q = self.tokenizer.decode(out[0], skip_special_tokens=True)
        q = q.replace("question:", "").strip()
        if len(q) < 6 or re.fullmatch(r'[\W_]+', q):
            return ""
        return q
