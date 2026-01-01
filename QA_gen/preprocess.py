from typing import List
import spacy
import re

def create_nlp():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp

nlp = create_nlp()

def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\b(subject|instructor|department|references|figure|fig\.|slide)\b[:\-]?[^\n]*', ' ', text, flags=re.IGNORECASE)  # Remove Metadata Lines
    text = re.sub(r'[•◦▪\-\*]+', '. ', text)  # Normalize Bullets and Dashes
    text = re.sub(r'\bnbsp\b|\&nbsp;|\&#160;', ' ', text, flags=re.IGNORECASE) # Replace HTML Non-breaking Spaces
    text = re.sub(r'[\u2022\u25CF\u25AA]+', ' ', text) # Replace Unicode Bullets
    text = re.sub(r'\n\s*\n', '. ', text) # Merge Double Newlines into Dots
    text = text.replace('\n', ' ') # Flatten All Lines to One Line
    text = re.sub(r'[.]{2,}', '. ', text) # Replace Multiple Dots & Spaces
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def _is_example_sentence(s: str) -> bool:
    s_low = s.lower()
    if 'consider the sentence' in s_low or s_low.startswith('example') or s_low.startswith('consider the') or s_low.startswith('the operation was') or s_low.startswith('the soldiers'):
        return True
    if re.match(r'^\d+[\.\)\-]\s*\w+', s.strip()) and len(s.split()) < 8:
        return True
    return False

def split_sentences(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    cleaned_text = _clean_text(text)
    doc = nlp(cleaned_text)
    sents = []
    for sent in doc.sents:  
        s = sent.text.strip()
        if len(s) < 20:  # Skip very short sentences
            continue
        if not re.search(r'[a-zA-Z]', s):  # Skip non-alphabetic text
            continue
        if _is_example_sentence(s): # Skip non-alphabetic text
            continue
        sents.append(s)
    return sents
