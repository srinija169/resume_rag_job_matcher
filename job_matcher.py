"""
Hybrid resume/job matching.
This is a decision-support demo, not an automated hiring decision system.
Always review evidence and use human judgment.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
DB_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "resume_chunks"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SKILL_TERMS = [
    "Python", "SQL", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go",
    "Machine Learning", "Deep Learning", "NLP", "LLM", "PyTorch", "TensorFlow",
    "Scikit-learn", "Pandas", "NumPy", "Spark", "Hadoop", "AWS", "Azure", "GCP",
    "Docker", "Kubernetes", "Git", "FastAPI", "Flask", "Django", "React",
    "Node.js", "Power BI", "Tableau", "Excel", "Statistics", "Data Analysis",
    "ETL", "Airflow", "Databricks", "ChromaDB", "RAG", "LangChain"
]

def extract_skills(text: str) -> list[str]:
    return [skill for skill in SKILL_TERMS if re.search(r"(?<!\w)" + re.escape(skill) + r"(?!\w)", text, re.I)]

def required_years(job_description: str) -> int:
    patterns = [
        r"(\d{1,2})\s*\+\s*years?",
        r"minimum of\s+(\d{1,2})\s+years?",
        r"at least\s+(\d{1,2})\s+years?",
        r"(\d{1,2})\s+or more years?"
    ]
    found = []
    for pattern in patterns:
        found.extend(int(n) for n in re.findall(pattern, job_description, re.I))
    return max(found, default=0)

def get_collection():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError("No resume index found. Run `python resume_rag.py` first.") from exc

def match_job(job_description: str, top_k: int = 10, enforce_must_haves: bool = True) -> dict:
    start = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME)
    collection = get_collection()
    count = collection.count()
    if count == 0:
        raise RuntimeError("The resume index is empty. Add resumes and run `python resume_rag.py`.")
    query_embedding = model.encode([job_description], normalize_embeddings=True).tolist()[0]
    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(max(top_k * 5, top_k), count),
        include=["documents", "metadatas", "distances"],
    )

    jd_skills = extract_skills(job_description)
    min_years = required_years(job_description)
    candidates = {}
    docs = raw.get("documents", [[]])[0]
    metas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    for doc, meta, distance in zip(docs, metas, distances):
        path = meta.get("resume_path", "")
        key = path or meta.get("candidate_name", "Unknown")
        if key not in candidates:
            candidates[key] = {
                "candidate_name": meta.get("candidate_name", "Unknown"),
                "resume_path": path,
                "skills": set(),
                "experience_years": 0,
                "education": meta.get("education", "Not specified"),
                "sections": [],
                "excerpts": [],
                "semantic_scores": [],
                "full_skill_text": meta.get("skills", ""),
            }
        c = candidates[key]
        c["skills"].update(s.strip() for s in meta.get("skills", "").split(",") if s.strip())
        c["experience_years"] = max(c["experience_years"], int(meta.get("experience_years", 0) or 0))
        c["semantic_scores"].append(max(0.0, min(1.0, 1.0 - float(distance))))
        section = meta.get("section", "General")
        if section not in c["sections"]:
            c["sections"].append(section)
        if doc and doc not in c["excerpts"] and len(c["excerpts"]) < 4:
            c["excerpts"].append(doc[:500])

    results = []
    for c in candidates.values():
        resume_skills = set(c["skills"]) | set(extract_skills(" ".join(c["excerpts"])))
        matched_skills = sorted(set(jd_skills) & resume_skills)
        skill_score = len(matched_skills) / max(1, len(jd_skills))
        semantic_score = sum(c["semantic_scores"]) / max(1, len(c["semantic_scores"]))
        # A transparent heuristic score: semantic similarity 60%, skill overlap 30%, experience 10%.
        experience_score = 1.0 if min_years == 0 else min(1.0, c["experience_years"] / min_years)
        score = round(100 * (0.60 * semantic_score + 0.30 * skill_score + 0.10 * experience_score))
        missing_skills = sorted(set(jd_skills) - resume_skills)
        passes_years = c["experience_years"] >= min_years
        passes_skills = len(missing_skills) == 0
        eligible = passes_years and passes_skills
        reasoning_parts = [
            f"Semantic relevance: {semantic_score:.0%}.",
            f"Matched {len(matched_skills)} of {len(jd_skills)} detected job skills.",
            f"Experience extracted: {c['experience_years']} years; requirement detected: {min_years} years."
        ]
        if matched_skills:
            reasoning_parts.append("Evidence of matching skills: " + ", ".join(matched_skills) + ".")
        if missing_skills:
            reasoning_parts.append("Job skills not detected in this resume: " + ", ".join(missing_skills) + ".")
        reasoning_parts.append("Relevant sections: " + ", ".join(c["sections"]) + ".")
        results.append({
            "candidate_name": c["candidate_name"],
            "resume_path": c["resume_path"],
            "match_score": score,
            "matched_skills": matched_skills,
            "relevant_excerpts": c["excerpts"][:3],
            "reasoning": " ".join(reasoning_parts),
            "experience_years": c["experience_years"],
            "missing_detected_skills": missing_skills,
            "must_have_pass": eligible,
            "_semantic_score": semantic_score,
            "_skill_score": skill_score,
        })

    results.sort(
    key=lambda x: (
        x["must_have_pass"],
        x["match_score"],
        x["_skill_score"],
        x["_semantic_score"],
        x["experience_years"],
    ),
    reverse=True,
)
    if enforce_must_haves and min_years:
        # Keep non-qualifying candidates out when a clear years requirement was detected.
        results = [r for r in results if r["experience_years"] >= min_years]
    results = results[:top_k]
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    for item in results:
        item.pop("_semantic_score", None)
        item.pop("_skill_score", None)
    return {
        "job_description": job_description,
        "filters": {"minimum_experience_years_detected": min_years, "must_have_skills_detected": jd_skills},
        "top_matches": results,
        "metrics": {"latency_ms": latency_ms, "results_returned": len(results)},
        "scoring_note": "Heuristic demo score, not a calibrated probability of job performance. Review source excerpts; extraction may miss facts."
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Match a job description to resumes.")
    parser.add_argument("--jd", help="Job description text.")
    parser.add_argument("--jd-file", help="Path to a text file containing the job description.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--allow-nonqualifying", action="store_true", help="Do not filter by detected minimum years.")
    parser.add_argument("--output", help="Optional path to write JSON output.")
    args = parser.parse_args()
    if args.jd_file:
        jd = Path(args.jd_file).read_text(encoding="utf-8")
    elif args.jd:
        jd = args.jd
    else:
        jd = input("Paste the job description, then press Enter:\n")
    result = match_job(jd, top_k=args.top_k, enforce_must_haves=not args.allow_nonqualifying)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
        print(f"\nSaved results to {args.output}")

if __name__ == "__main__":
    main()
