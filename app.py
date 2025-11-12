import streamlit as st
from QA_gen.extractor import extract_text_from_file
from QA_gen.preprocess import split_sentences
from QA_gen.answer_selector import select_candidate_answers
from QA_gen.ques_model import QuestionGenerator
from QA_gen.distractors import generate_distractors_for_answer
import pandas as pd
import io
import re

import logging
import os
os.environ["WANDB_DISABLED"] = "true"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

@st.cache_resource
def load_models(model_name="t5-small"):
    """
    Load required models.
    Returns dict of models or raises a friendly exception.
    """
    import nltk
    import spacy
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from sentence_transformers import SentenceTransformer
    model_id = model_name
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    try:
        sbert = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        sbert = None

    return {"nlp": nlp, "tokenizer": tokenizer, "model": model, "sbert": sbert}

def assign_difficulty(question, answer, context, is_fallback):
    """
    Assign a heuristic-based difficulty level.
    """
    question = question.lower()
    
    if is_fallback:
        return "Easy"
    
    if question.startswith("why") or question.startswith("how"):
        return "Hard"

    if len(answer.split()) <= 2:
        return "Easy"
    
    if len(context.split()) > 40:
        return "Medium"

    return "Medium"


st.set_page_config(page_title="QA & MCQ Generator", layout="wide")

st.title("QA & MCQ Generator")
st.markdown("Upload a text/docx/pdf/csv and generate MCQs.")

uploaded_file = st.file_uploader("Upload file (txt/docx/pdf/csv)", type=["txt","docx","pdf","csv"])

skip_pages = st.checkbox("For PDFs, skip first and last page (to avoid cover/reference pages)", value=True)

num_questions = st.slider("Max questions to generate", 1, 40, 10)
use_keybert = st.checkbox("Use KeyBERT for keywords (better seeds)", value=True)

allowed_models = ["t5-small", "t5-base", "google/flan-t5-small"]
model_size = st.selectbox("QG model (FLAN is best)", allowed_models, index=2) 

st.warning("For best results, use the **google/flan-t5-small** model and check the **KeyBERT** box.")


if "mcq_list" not in st.session_state:
    st.session_state.mcq_list = []
if "mcq_submitted" not in st.session_state:
    st.session_state.mcq_submitted = {}


if uploaded_file and st.session_state.mcq_list and st.session_state.get("_last_uploaded_name") != uploaded_file.name:
    st.session_state.mcq_list = []
    st.session_state.mcq_submitted = {}

def fallback_cloze_question(sentence: str, answer: str) -> str:
    """
    Create a simple fill-in-the-blank question by masking the answer in the sentence.
    """
    sent_lower = sentence.lower()
    ans_lower = answer.lower().strip()
    if ans_lower and ans_lower in sent_lower:
        pattern = re.compile(re.escape(answer), flags=re.IGNORECASE)
        masked = pattern.sub("_____", sentence, count=1)
        return f"Fill in the blank: {masked}"
    else:
        short = (answer[:80] + "...") if len(answer) > 80 else answer
        return f"What is {short}?"

if uploaded_file:
    with st.spinner("Extracting text..."):
        raw_text = extract_text_from_file(uploaded_file, skip_cover_pages=skip_pages)
    st.success("Text extracted")

    st.session_state["_last_uploaded_name"] = uploaded_file.name

    if st.checkbox("Show preview of uploaded text (first 1000 chars)"):
        st.write("Preview (first 1000 chars):")
        st.code(raw_text[:1000])

    if st.button("Prepare & Generate"):
        st.session_state.mcq_list = []
        st.session_state.mcq_submitted = {}

        with st.spinner("Preparing document and selecting answer candidates..."):
            sentences = split_sentences(raw_text)
            sentences = [s for i,s in enumerate(sentences) if s and len(s.split()) >= 4]
            sentences.sort(key=lambda x: -len(x))
            candidates = select_candidate_answers(sentences, use_keybert=use_keybert, top_k=300)
            filtered = []
            for sent, ans in candidates:
                if len(ans.split()) > 0 and len(ans) >= 3 and not ans.lower().strip() in {'lstm','rnn','◦'}:
                    filtered.append((sent, ans))
            candidates = filtered


        if not candidates:
            st.warning("No candidate answers found in the document. Try a different document or enable KeyBERT keyword extraction.")
        else:
            with st.spinner("Generating questions..."):
                try:
                    models = load_models(model_size)
                except Exception as e:
                    st.warning(f"Could not load QG model {model_size}. Using fallback cloze-only generation. (Error: {e})")
                    models = None
                qg = None
                if models:
                    qg = QuestionGenerator(
                        tokenizer=models["tokenizer"], 
                        model=models["model"]
                    )

                from random import shuffle
                generated = []
                seen_sentences = set()

                for sent, answer in candidates:
                    if sent in seen_sentences:
                        continue
                        
                    q_text = ""
                    is_fallback = False
                    
                    if qg is not None: # Check if 'qg' was successfully created
                        try:
                            q_text = qg.generate_question(context=sent, answer=answer)
                        except Exception:
                            q_text = "" 
                    q_text_lower = q_text.lower().strip()
                    ans_lower = answer.lower().strip()
                    
                    is_garbage = (
                        not q_text 
                        or len(q_text_lower) <= 5 
                        or q_text_lower == "true" 
                        or not re.search(r'[a-zA-Z]', q_text)
                        or ans_lower in q_text_lower  # Answer is IN the question
                        or (len(ans_lower.split()) > 1 and q_text_lower in ans_lower) # Question is IN the answer
                    )

                    if is_garbage:
                        q_text = fallback_cloze_question(sent, answer)
                        is_fallback = True 

                    try:
                        distractors = generate_distractors_for_answer(answer, raw_text, top_k=3)
                    except Exception as e:
                        print(f"Distractor error: {e}")
                        distractors = []
                        
                    difficulty = assign_difficulty(q_text, answer, sent, is_fallback)

                    opts = [answer] + distractors
                    opts = list(dict.fromkeys([o.strip() for o in opts if o and o.strip()])) # Dedup

                    while len(opts) < 4:
                        opts.append("None of these")

                    shuffle(opts)
                    correct_idx = opts.index(answer) if answer in opts else 0

                    generated.append({
                        "question": q_text,
                        "answer": answer,
                        "options": opts,
                        "correct_idx": correct_idx,
                        "context": sent,
                        "difficulty": difficulty 
                    })
                    seen_sentences.add(sent)
                    if len(generated) >= num_questions:
                        break

                if not generated:
                    st.error("No questions were generated. Try a different file or enable KeyBERT.")
                else:
                    st.session_state.mcq_list = generated

# Display MCQs
if st.session_state.mcq_list:
    st.markdown("---")
    st.subheader("Attempt the MCQs")

    for i, mcq in enumerate(st.session_state.mcq_list):
        qid = f"mcq_{i}"
        st.markdown(f"**Q{i+1}.** [{mcq['difficulty']}] {mcq['question']}")
        
        choice = st.radio(label=f"Select answer for Q{i+1}", options=mcq["options"], key=qid, index=None)
        
        try:
            st.session_state.mcq_submitted[qid] = mcq["options"].index(choice)
        except ValueError:
            
            st.session_state.mcq_submitted[qid] = None

        selected_idx = st.session_state.mcq_submitted.get(qid, None)
        
        if selected_idx is not None:
            if selected_idx == mcq["correct_idx"]:
                st.success("Correct answer")
            else:
                correct_text = mcq["options"][mcq["correct_idx"]]
                st.error(f"Wrong answer, the correct answer is **{correct_text}**")

        with st.expander("Show context sentence"):
            st.write(mcq["context"])

        st.markdown("---")

    cols = st.columns([1,1,2])
    if cols[0].button("Regenerate questions"):
        st.session_state.mcq_list = []
        st.session_state.mcq_submitted = {}
        st.rerun()

    # Download CSV export
    df_rows = []
    for idx, m in enumerate(st.session_state.mcq_list):
        df_rows.append({
            "question_no": idx+1,
            "question": m["question"],
            "answer": m["answer"],
            "options": "||".join(m["options"]),
            "context": m["context"],
            "difficulty": m["difficulty"]
        })
    csv_buf = io.StringIO()
    pd.DataFrame(df_rows).to_csv(csv_buf, index=False)
    st.download_button("Download questions as CSV", csv_buf.getvalue(), "questions.csv", "text/csv")

else:
    st.info("No questions to attempt yet. Upload a file and click 'Prepare & Generate' to create MCQs.")