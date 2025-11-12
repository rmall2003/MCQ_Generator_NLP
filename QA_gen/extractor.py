from io import BytesIO
import docx
import PyPDF2
import pandas as pd
import re

def _normalize_page_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace('\xa0', ' ').replace('&nbsp;', ' ').replace('nbsp;', ' ')
    text = re.sub(r'[•◦▪●♦·•►]+', ' • ', text)
    text = re.sub(r'\.{2,}', '. ', text)
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line: 
            continue
        if re.fullmatch(r'page\s*\d+|\d+', line.lower()):
            continue
        lines.append(line)
    return " ".join(lines)

def extract_text_from_file(uploaded_file, skip_cover_pages=True):
    fname = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if fname.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")
    elif fname.endswith(".docx"):
        doc = docx.Document(BytesIO(data))
        paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        return "\n".join(paragraphs)
    elif fname.endswith(".pdf"):
        reader = PyPDF2.PdfReader(BytesIO(data))
        text_pages = []
        num_pages = len(reader.pages)

        # Skip first/last if requested and file has >2 pages
        if skip_cover_pages and num_pages > 2:
            start_page, end_page = 1, num_pages - 1
        else:
            start_page, end_page = 0, num_pages

        for i in range(start_page, end_page):
            try:
                page = reader.pages[i]
                text = page.extract_text() or ""
                normalized = _normalize_page_text(text)
                if normalized:
                    text_pages.append(normalized)
            except Exception:
                continue
        return "\n".join(text_pages)
    elif fname.endswith(".csv"):
        df = pd.read_csv(BytesIO(data))
        return "\n".join(df.astype(str).agg(" | ".join, axis=1).tolist())
    else:
        raise ValueError("Unsupported file type")
