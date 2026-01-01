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

# Disable Unnecessary Logging (suppress warnings from Hugging Face and SentenceTransformer)
os.environ["WANDB_DISABLED"] = "true"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

@st.cache_resource
def load_models(model_name="google/flan-t5-base"):
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

# Custom CSS for a dark theme
st.markdown("""
    <style>
        .main { padding-top: 2rem; }
        .stApp { background-color: #0f172a; }
        h1 { color: #f1f5f9; font-weight: 600; margin-bottom: 0.5rem; }
        .subtitle { color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem; }
        .control-section { 
            background: #1e293b; 
            padding: 1.5rem; 
            border-radius: 10px; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            margin-bottom: 2rem;
            border: 1px solid #334155;
        }
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.5rem 2rem;
            font-weight: 500;
            border-radius: 6px;
            transition: transform 0.2s;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.6);
        }
        .stDownloadButton>button {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            border: none;
            padding: 0.5rem 2rem;
            font-weight: 500;
            border-radius: 6px;
        }
        .mcq-card {
            background: #1e293b;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            margin-bottom: 1.5rem;
            border-left: 4px solid #667eea;
            border: 1px solid #334155;
        }
        .difficulty-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-left: 0.5rem;
        }
        .difficulty-easy { background-color: #064e3b; color: #6ee7b7; }
        .difficulty-medium { background-color: #78350f; color: #fcd34d; }
        .difficulty-hard { background-color: #7f1d1d; color: #fca5a5; }
        
        /* Dark theme adjustments for Streamlit components */
        .stRadio > label { color: #e2e8f0 !important; }
        .stExpander { background-color: #1e293b; border: 1px solid #334155; }
        [data-testid="stExpander"] { background-color: #1e293b; }
        .stMarkdown { color: #e2e8f0; }
    </style>
""", unsafe_allow_html=True)

st.title("🎓 QA & MCQ Generator")
st.markdown('<p class="subtitle">Upload your document and generate intelligent multiple-choice questions</p>', unsafe_allow_html=True)

# Control section with cleaner layout
st.markdown('<div class="control-section">', unsafe_allow_html=True)
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader("📁 Upload your document", type=["txt","docx","pdf","csv"], help="Supports text, Word, PDF, and CSV files")

with col2:
    num_questions = st.slider("Number of questions", 1, 40, 10, help="Maximum questions to generate")

col3, col4 = st.columns(2)
with col3:
    use_keybert = st.checkbox("✨ Use KeyBERT for better keywords", value=True, help="Improves question quality by extracting key phrases")
with col4:
    skip_pages = st.checkbox("📄 Skip first & last PDF pages", value=True, help="Useful to avoid cover and reference pages")

st.markdown('</div>', unsafe_allow_html=True)

model_size = "google/flan-t5-base"


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

def clean_question(q_text: str, answer: str) -> str:
    """
    Post-process and validate generated questions for quality.
    Returns empty string if question is invalid.
    """
    if not q_text:
        return ""
    
    q_text = q_text.strip()
    
    # Remove answer from question if present (case-insensitive check)
    ans_lower = answer.lower()
    if ans_lower in q_text.lower():
        return ""
    
    # Remove common prefixes
    prefixes = ["question:", "q:", "answer:", "a:"]
    for prefix in prefixes:
        if q_text.lower().startswith(prefix):
            q_text = q_text[len(prefix):].strip()
    
    # Ensure it ends with a question mark
    if q_text and not q_text.endswith('?'):
        q_text += '?'
    
    # Capitalize first letter
    if q_text and len(q_text) > 0:
        q_text = q_text[0].upper() + q_text[1:]
    
    # Validate minimum length and content
    if len(q_text) < 10:
        return ""
    
    # Check if it has actual words (not just symbols)
    if not re.search(r'[a-zA-Z]{3,}', q_text):
        return ""
    
    # Avoid questions that are just "What is X?" where X is too short
    if re.match(r'^What is .{1,15}\?$', q_text, re.IGNORECASE):
        return ""
    
    return q_text

if uploaded_file:
    with st.spinner("📖 Extracting text from your document..."):
        raw_text = extract_text_from_file(uploaded_file, skip_cover_pages=skip_pages)
    st.success("✅ Text extracted successfully!")

    st.session_state["_last_uploaded_name"] = uploaded_file.name

    with st.expander("👁️ Preview extracted text (first 1000 characters)"):
        st.code(raw_text[:1000], language="text")

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 Generate MCQs", use_container_width=True):
        st.session_state.mcq_list = []
        st.session_state.mcq_submitted = {}

        with st.spinner("🔍 Preparing document and selecting answer candidates..."):
            sentences = split_sentences(raw_text)
            sentences = [s for i,s in enumerate(sentences) if s and len(s.split()) >= 4]
            sentences.sort(key=lambda x: -len(x))
            candidates = select_candidate_answers(sentences, use_keybert=use_keybert, top_k=300)
            
            # Enhanced answer filtering for better quality
            filtered = []
            stopwords = {'lstm', 'rnn', '◦', 'etc', 'e.g', 'i.e', 'e.g.', 'i.e.', 'vs', 'vs.'}
            
            for sent, ans in candidates:
                ans_stripped = ans.strip()
                ans_words = ans_stripped.split()
                ans_lower = ans_stripped.lower()
                
                # Enhanced filtering criteria
                is_valid = (
                    len(ans_words) >= 2 and len(ans_words) <= 10 and  # Not too short or long
                    len(ans_stripped) >= 5 and len(ans_stripped) <= 100 and  # Character limits
                    ans_lower not in stopwords and  # Not a stopword
                    not ans_stripped.startswith(('•', '-', '◦', '*', '(', '[')) and  # No bullet points
                    not ans_stripped.endswith((':', ',', ';')) and  # No trailing punctuation
                    ans_stripped[0].isupper() and  # Starts with capital
                    any(c.isalpha() for c in ans_stripped) and  # Contains letters
                    not ans_stripped.isnumeric() and  # Not just a number
                    sum(c.isdigit() for c in ans_stripped) / len(ans_stripped) < 0.5  # Not mostly digits
                )
                
                if is_valid:
                    filtered.append((sent, ans_stripped))
            
            # Get more candidates to ensure quality after filtering
            candidates = filtered[:num_questions * 3]


        if not candidates:
            st.warning("⚠️ No candidate answers found. Try a different document or enable KeyBERT keyword extraction.")
        else:
            with st.spinner("🤖 Generating questions using AI..."):
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
                            # Apply post-processing to clean and validate the question
                            q_text = clean_question(q_text, answer)
                        except Exception:
                            q_text = "" 
                    
                    # If question is still invalid after cleaning, use fallback
                    if not q_text or len(q_text) < 10:
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
                    st.error("❌ No questions were generated. Try a different file or enable KeyBERT.")
                else:
                    st.session_state.mcq_list = generated
                    st.success(f"✅ Successfully generated {len(generated)} questions!")

# Display MCQs
if st.session_state.mcq_list:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## 📝 Your Generated MCQs")
    st.markdown(f"*Answer the questions below. Total: {len(st.session_state.mcq_list)} questions*")
    st.markdown("<br>", unsafe_allow_html=True)

    for i, mcq in enumerate(st.session_state.mcq_list):
        qid = f"mcq_{i}"
        
        # Difficulty badge HTML
        difficulty = mcq['difficulty'].lower()
        badge_class = f"difficulty-{difficulty}"
        
        st.markdown('<div class="mcq-card">', unsafe_allow_html=True)
        st.markdown(f"""
            <div style="margin-bottom: 1rem;">
                <span style="font-size: 1.1rem; font-weight: 600; color: #f1f5f9;">Q{i+1}.</span>
                <span class="difficulty-badge {badge_class}">{mcq['difficulty']}</span>
            </div>
            <div style="font-size: 1.05rem; color: #cbd5e1; margin-bottom: 1rem;">{mcq['question']}</div>
        """, unsafe_allow_html=True)
        
        choice = st.radio(
            label=f"Select your answer for Q{i+1}", 
            options=mcq["options"], 
            key=qid, 
            index=None,
            label_visibility="collapsed"
        )
        
        try:
            st.session_state.mcq_submitted[qid] = mcq["options"].index(choice)
        except ValueError:
            st.session_state.mcq_submitted[qid] = None

        selected_idx = st.session_state.mcq_submitted.get(qid, None)
        
        if selected_idx is not None:
            if selected_idx == mcq["correct_idx"]:
                st.success("✅ Correct! Well done!")
            else:
                correct_text = mcq["options"][mcq["correct_idx"]]
                st.error(f"❌ Incorrect. The correct answer is: **{correct_text}**")

        with st.expander("💡 Show context"):
            st.info(mcq["context"])
        
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("🔄 Regenerate Questions"):
            st.session_state.mcq_list = []
            st.session_state.mcq_submitted = {}
            st.rerun()
    
    with col2:
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
        st.download_button("📥 Download CSV", csv_buf.getvalue(), "mcq_questions.csv", "text/csv")

else:
    st.info("👆 Upload a document above and click 'Generate MCQs' to get started!")