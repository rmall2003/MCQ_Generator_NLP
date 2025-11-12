from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import re

try:
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Error loading SBERT: {e}. Distractors may be weaker.")
    sbert = None

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def wn_synonyms(word):
    syns = set()
    try:
        for syn in wn.synsets(word):
            for l in syn.lemmas():
                name = l.name().replace("_", " ")
                if name.lower() != word.lower():
                    syns.add(name)
    except Exception:
        pass
    return list(syns)

def build_dynamic_pool(text: str, min_len=3, max_size=1000) -> list:
    doc = nlp(text)
    pool = set()
    for chunk in doc.noun_chunks:
        cand = chunk.text.strip()
        if len(cand) >= min_len and not cand.lower() in {"this","that","these","those","it","they"}:
            if not re.fullmatch(r'\d+|\W+', cand):
                pool.add(cand)
    if not pool:
        for token in doc:
            if token.pos_ == "NOUN" and len(token.text) >= min_len and not token.is_stop:
                pool.add(token.text.strip())
    pool_list = list(pool)
    np.random.shuffle(pool_list)
    return pool_list[:max_size]

def generate_distractors_for_answer(answer: str, document_text: str, top_k=3):
    ans = answer.strip()
    tokens = ans.split()
    distractors = []
    if sbert is not None:
        pool = build_dynamic_pool(document_text)
        pool = [p for p in pool if p.lower() != ans.lower() and len(p) > 2]
        if pool:
            try:
                pool_emb = sbert.encode(pool)
                ans_emb = sbert.encode([ans])[0]
                sims = cosine_similarity([ans_emb], pool_emb)[0]
                idxs = np.argsort(-sims)
                for i in idxs:
                    cand = pool[i]
                    if sims[i] > 0.4 and cand.lower() != ans.lower() and cand not in distractors:
                        distractors.append(cand)
                    if len(distractors) >= top_k:
                        break
            except Exception:
                pass
    if len(distractors) < top_k and len(tokens) == 1:
        syns = wn_synonyms(tokens[0])
        for s in syns:
            if s.lower() != ans.lower() and s not in distractors:
                distractors.append(s)
                if len(distractors) >= top_k:
                    return distractors[:top_k]
    idx = 1
    while len(distractors) < top_k:
        candidate = f"Option {idx+1}" 
        if candidate not in distractors and candidate.lower() != ans.lower():
            distractors.append(candidate)
        idx += 1
        
    return distractors[:top_k]
