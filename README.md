# MCQ Generator (NLP-Powered) 📝🤖

An intelligent Multiple Choice Question (MCQ) generator powered by Natural Language Processing. This tool automatically generates high-quality MCQs from text documents, PDFs, or raw text input using advanced NLP models and transformers.

## ✨ Features

- **Multiple Input Formats**: Upload PDF, DOCX, TXT files or paste text directly
- **Automatic Question Generation**: Uses Google's FLAN-T5 model for question generation
- **Smart Answer Selection**: Leverages spaCy NER to identify key entities and concepts
- **Distractor Generation**: Creates plausible wrong answers using:
  - WordNet semantic similarity
  - Sentence embeddings (all-MiniLM-L6-v2)
- **Customizable Settings**:
  - Number of questions to generate
  - Number of answer options per question
  - Model selection
- **Export Options**: Download generated MCQs in various formats
- **Interactive UI**: Built with Streamlit for easy use

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **NLP Models**:
  - FLAN-T5 (google/flan-t5-base) for question generation
  - spaCy (en_core_web_sm) for NER and text processing
  - SentenceTransformers (all-MiniLM-L6-v2) for semantic similarity
- **Text Processing**:
  - PyPDF2/pdfminer for PDF extraction
  - python-docx for DOCX files
- **Libraries**: NLTK, transformers, sentence-transformers

## 📋 Prerequisites

- Python 3.8 or higher
- ~2GB free disk space (for model downloads)

## 🚀 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/rmall2003/MCQ_Generator_NLP
cd MCQ_Generator_NLP
```

### 2. Create virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install spaCy language model

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

### 5. Run the application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 Usage

1. **Input Text**:
   - Upload a file (PDF, DOCX, TXT), or
   - Paste text directly into the text area
2. **Configure Settings**:
   - Number of questions to generate
   - Number of answer choices per question
   - Select transformer model (if multiple options available)
3. **Generate MCQs**: Click the generate button
4. **Review & Export**: Review generated questions and download as needed

## 📁 Project Structure

```
MCQ_Generator_NLP/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── README.md                # This file
└── QA_gen/                  # Core MCQ generation modules
    ├── __init__.py
    ├── extractor.py         # Text extraction from files
    ├── preprocess.py        # Text preprocessing and sentence splitting
    ├── answer_selector.py   # Candidate answer selection using NER
    ├── ques_model.py        # Question generation using T5
    ├── distractors.py       # Wrong answer generation
    └── utlis.py             # Utility functions
```

## 🔧 Key Components

### Text Extraction (`extractor.py`)

Handles extraction from multiple file formats (PDF, DOCX, TXT).

### Preprocessing (`preprocess.py`)

Splits text into sentences and prepares it for analysis.

### Answer Selection (`answer_selector.py`)

Uses spaCy's Named Entity Recognition to identify important concepts as candidate answers.

### Question Generation (`ques_model.py`)

Leverages FLAN-T5 transformer model to generate contextual questions.

### Distractor Generation (`distractors.py`)

Creates plausible wrong answers using:

- WordNet for semantic relationships
- Sentence embeddings for similarity scoring

## ⚙️ Model Configuration

The default configuration uses:

- **Question Generation**: `google/flan-t5-base`
- **Sentence Embeddings**: `all-MiniLM-L6-v2`
- **NER**: spaCy `en_core_web_sm`

Models are cached after first download for faster subsequent loads.

## 🐛 Troubleshooting

### spaCy Model Missing

If you get "Can't find model 'en_core_web_sm'", run:

```bash
python -m spacy download en_core_web_sm
```

### Slow Generation

Question generation is compute-intensive. Expected times:

- First run: 2-5 minutes (model loading)
- Subsequent runs: 30-60 seconds per question
