"""
Resume RAG ingestion pipeline.
- Reads PDF, TXT, and DOCX resumes from ./resumes
- Splits resumes by section headings where possible
- Extracts basic metadata
- Embeds chunks with a local Sentence Transformers model
- Persists vectors in ChromaDB
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
RESUME_DIR = ROOT / "resumes"
DB_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "resume_chunks"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SECTION_NAMES = {
    "summary": "Summary", "professional summary": "Summary", "profile": "Summary",
    "skills": "Skills", "technical skills": "Skills", "core competencies": "Skills",
    "experience": "Experience", "work experience": "Experience",
    "professional experience": "Experience", "employment history": "Experience",
    "education": "Education", "certifications": "Certifications",
    "projects": "Projects", "personal projects": "Projects",
}

def read_resume(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".txt" or suffix == ".md":
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".docx":
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError("For .docx files install python-docx: pip install python-docx") from exc
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError(f"Unsupported resume format: {path.suffix}")

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def extract_metadata(text: str, path: Path) -> dict:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    name = path.stem.replace("_", " ").replace("-", " ").title()
    if lines:
        candidate = lines[0]
        if len(candidate) <= 60 and not re.search(r"@|https?://|resume|curriculum vitae", candidate, re.I):
            if not re.search(r"\d{3,}", candidate):
                name = candidate
    skill_vocab = [
        "Python", "SQL", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go",
        "Machine Learning", "Deep Learning", "NLP", "LLM", "PyTorch", "TensorFlow",
        "Scikit-learn", "Pandas", "NumPy", "Spark", "Hadoop", "AWS", "Azure", "GCP",
        "Docker", "Kubernetes", "Git", "FastAPI", "Flask", "Django", "React",
        "Node.js", "Power BI", "Tableau", "Excel", "Statistics", "Data Analysis",
        "ETL", "Airflow", "Databricks", "ChromaDB", "RAG", "LangChain"
    ]
    skills = [s for s in skill_vocab if re.search(r"(?<!\w)" + re.escape(s) + r"(?!\w)", text, re.I)]
    year_values = [int(x) for x in re.findall(r"(\d{1,2})\+?\s+years?", text, re.I)]
    experience_years = max(year_values, default=0)
    education = []
    for line in lines:
        if re.search(r"\b(B\.?S\.?|B\.?E\.?|B\.?Tech|M\.?S\.?|M\.?Tech|MBA|Ph\.?D|Bachelor|Master|BSc|MSc)\b", line, re.I):
            education.append(line[:180])
    return {
        "candidate_name": name,
        "skills": skills,
        "experience_years": experience_years,
        "education": " | ".join(education[:3]) or "Not specified",
        "resume_path": str(path.relative_to(ROOT)).replace("\\", "/"),
    }

def chunk_resume(text: str, max_chars: int = 1100, overlap: int = 120) -> list[dict]:
    """Split by recognized section headings first, then by character length."""
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_name = "General"
    current_lines: list[str] = []

    def flush():
        nonlocal current_lines
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_name, current_lines.copy()))
        current_lines = []

    for line in lines:
        stripped = line.strip().strip(":").strip()
        normalized = re.sub(r"\s+", " ", stripped).lower()
        if normalized in SECTION_NAMES:
            flush()
            current_name = SECTION_NAMES[normalized]
        else:
            current_lines.append(line)
    flush()

    # If headings weren't recognized, preserve the document as a General section.
    if not sections:
        sections = [("General", lines)]

    chunks = []
    for section, section_lines in sections:
        body = "\n".join(section_lines).strip()
        if not body:
            continue
        paragraphs = re.split(r"\n\s*\n", body)
        current = ""
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if len(current) + len(paragraph) + 1 <= max_chars:
                current = f"{current}\n{paragraph}".strip()
                continue
            if current:
                chunks.append({"section": section, "text": current})
            if len(paragraph) > max_chars:
                start = 0
                while start < len(paragraph):
                    piece = paragraph[start:start + max_chars]
                    if piece.strip():
                        chunks.append({"section": section, "text": piece})
                    start += max_chars - overlap
                current = ""
            else:
                current = paragraph
        if current:
            chunks.append({"section": section, "text": current})
    return chunks

def build_index(reset: bool = False) -> int:
    if not RESUME_DIR.exists():
        raise FileNotFoundError(f"Resume folder not found: {RESUME_DIR}")
    client = chromadb.PersistentClient(path=str(DB_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    model = SentenceTransformer(MODEL_NAME)
    paths = sorted(p for p in RESUME_DIR.rglob("*") if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md", ".docx"})
    if not paths:
        print(f"No resumes found in {RESUME_DIR}. Add PDF/TXT/DOCX files and run again.")
        return 0

    ids, documents, metadatas = [], [], []
    for path in paths:
        try:
            text = clean_text(read_resume(path))
            if not text:
                print(f"Skipping empty/unreadable resume: {path.name}")
                continue
            meta = extract_metadata(text, path)
            chunks = chunk_resume(text)
            for i, chunk in enumerate(chunks):
                ids.append(f"{path.stem}-{i}")
                documents.append(chunk["text"])
                metadatas.append({
                    "candidate_name": meta["candidate_name"],
                    "skills": ", ".join(meta["skills"]),
                    "experience_years": int(meta["experience_years"]),
                    "education": meta["education"],
                    "resume_path": meta["resume_path"],
                    "section": chunk["section"],
                })
        except Exception as exc:
            print(f"Could not process {path.name}: {exc}")

    if not documents:
        return 0

    embeddings = model.encode(documents, normalize_embeddings=True, show_progress_bar=True).tolist()
    # Upsert makes rerunning ingestion safe for the same chunk IDs.
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    print(f"Indexed {len(paths)} resume files into {len(documents)} chunks.")
    print(f"ChromaDB directory: {DB_DIR}")
    return len(documents)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Index resumes into ChromaDB.")
    parser.add_argument("--reset", action="store_true", help="Delete the existing collection before indexing.")
    args = parser.parse_args()
    build_index(reset=args.reset)
