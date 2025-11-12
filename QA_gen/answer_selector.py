from typing import List, Tuple
import spacy
from keybert import KeyBERT
import re

nlp = spacy.load("en_core_web_sm")

def noun_phrases_from_sentence(sent):
    doc = nlp(sent)
    nps = [chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.strip())>2]
    return nps

def _clean_candidate_text(a: str) -> str:
    # Remove leftover bullets, headers and leading/trailing punctuation
    a = re.sub(r'[•◦▪\*]+', ' ', a)
    a = a.strip()
    a = re.sub(r'^[^A-Za-z0-9]+|[^A-Za-z0-9]+$', '', a)
    return a.strip()

def select_candidate_answers(sentences: List[str], use_keybert=True, top_k=100) -> List[Tuple[str,str]]:
    candidates = []
    kw_model = None
    if use_keybert:
        try:
            kw_model = KeyBERT()
        except Exception:
            kw_model = None

    for sent in sentences:
        nps = noun_phrases_from_sentence(sent)
        kws = []
        if use_keybert and kw_model:
            try:
                raw_kws = kw_model.extract_keywords(sent, top_n=5)
                # raw_kws returns list of (kw, score)
                kws = [kw for kw, _ in raw_kws if isinstance(kw, str)]
            except Exception:
                kws = []
        # prefer noun phrases first, then keybert keywords (if they are meaningful)
        answers = nps + kws
        seen = set()
        for a in answers:
            a = _clean_candidate_text(a)
            if len(a) < 3:
                continue
            # skip if candidate looks like a heading or starts/ends with digits
            if re.search(r'^[\d\W]+$', a) or re.match(r'^\d', a):
                continue
            if a.lower() in seen:
                continue
            seen.add(a.lower())
            candidates.append((sent, a))
            if len(candidates) >= top_k:
                return candidates
    return candidates
