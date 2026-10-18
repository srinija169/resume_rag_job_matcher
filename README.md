# Resume RAG & Job Matching Engine

A learning project that ingests resumes, chunks them by section, embeds chunks using a local Sentence Transformers model, stores vectors in ChromaDB, and ranks candidates using a hybrid semantic similarity and skill-overlap score.

The project evaluates candidate rankings across five job roles: Data Scientist, Data Engineer, BI Analyst, ML Engineer, and RAG Engineer.

## Important Notice

This is an educational decision-support prototype. Its scores must not be used as the sole basis for hiring decisions.

Resume parsing, skill detection, and years-of-experience extraction are heuristic and can be incorrect. Candidates should be reviewed using resume evidence and consistent, job-related criteria.

The current evaluation labels are provisional and require human verification before the reported metrics can be considered reliable.

## Features

- Resume loading and section-based chunking
- Local text embeddings using Sentence Transformers
- Vector storage and semantic retrieval with ChromaDB
- Hybrid candidate ranking using semantic similarity, skill overlap, and experience fit
- Extraction of detected skills and missing skills
- Minimum-experience filtering
- JSON output for ranked candidates
- Notebook experiments and latency analysis
- Precision@10 evaluation across five job roles
- CSV export of evaluation results

## Technology Stack

- Python 3.10 or 3.11 recommended
- Pandas
- Sentence Transformers
- ChromaDB
- Jupyter Notebook
- JSON and CSV for storing results

No paid API key is required.

## Requirements

- Python 3.10 or 3.11
- VS Code recommended
- Internet access for the initial installation and embedding-model download

## 1. Set Up the Environment

Open a terminal in the project folder.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.11` is unavailable, install Python 3.11 from [python.org](https://www.python.org/downloads/) and enable **Add Python to PATH** during installation.

If PowerShell blocks virtual-environment activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Index the Sample Resumes

Run:

```bash
python resume_rag.py --reset
```

The first run downloads the embedding model and may take a few minutes. The vector database is stored locally in `chroma_db/`.

**Warning:** The `--reset` option rebuilds the vector index. Do not run it unnecessarily if you want to preserve your existing index.

## 3. Match a Job Description

Run the matcher against a sample Data Scientist job description:

```bash
python job_matcher.py --jd-file job_descriptions/data_scientist.txt --top-k 10 --output results.json
```

To enter a job description interactively:

```bash
python job_matcher.py
```

To include candidates who do not meet the detected minimum-experience requirement:

```bash
python job_matcher.py --jd-file job_descriptions/data_scientist.txt --allow-nonqualifying
```

Use the actual command-line options supported by your installed script if they differ from these examples.

## 4. Run the Notebook

Start Jupyter:

```bash
jupyter notebook
```

Open `notebooks/experiments.ipynb` and run the cells from top to bottom.

The notebook contains experiments for candidate matching and evaluation. It uses the generated rankings and relevance labels to calculate Precision@10.

## Project Structure

```text
resume_rag_job_matcher/
├── resume_rag.py
├── job_matcher.py
├── resumes/
├── job_descriptions/
├── notebooks/
│   └── experiments.ipynb
├── chroma_db/
├── results.json
├── data_engineer_results.json
├── bi_results.json
├── ml_results.json
├── rag_results.json
├── evaluation_results.csv
├── requirements.txt
└── README.md
```

Some generated files may appear only after running the relevant scripts or notebook cells.

## Current Scoring Method

The current baseline scoring method is:

- **60%** semantic similarity
- **30%** detected skill overlap
- **10%** experience requirement fit

The score is a ranking heuristic, not a probability of being qualified or a calibrated measure of hiring suitability.

The ranking can be improved by evaluating it against human-reviewed relevance judgments and testing different scoring weights.

## Evaluation Method

### What is Precision@10?

Precision@10 measures the proportion of the top 10 ranked candidates that have been labeled relevant for a given job.

\[
\text{Precision@10} =
\frac{\text{Relevant candidates in top 10}}{10}
\]

For example, if 3 of the top 10 candidates are labeled relevant, Precision@10 is 0.30, or 30%.

### Current Evaluation Results

| Job role | Relevant in top 10 | Precision@10 |
|---|---:|---:|
| Data Scientist | 3/10 | 30% |
| Data Engineer | 3/10 | 30% |
| BI Analyst | 4/10 | 40% |
| ML Engineer | 4/10 | 40% |
| RAG Engineer | 2/10 | 20% |
| **Overall** | **16/50** | **32%** |

**Current average Precision@10: 32%.**

These results are based on the labels currently recorded in the notebook. The labels have not all been independently verified against complete resumes and job descriptions, so the metrics should be treated as preliminary rather than definitive.

### What Does the 32% Tell Us?

Under the current labels, 16 of the 50 candidates across the five top-10 lists were marked relevant. The other 34 were marked irrelevant.

This does not necessarily mean that the remaining 34 candidates are genuinely unsuitable. Some could be relevant but ranked too low, and some labels may be incorrect.

Potential causes of low measured precision include:

1. **Provisional relevance labels:** Some candidates were labeled using detected skills rather than a full review of their resumes against each job description.
2. **Incomplete skill vocabulary:** The parser may miss synonyms, related technologies, or equivalent experience.
3. **Semantic and skill-score balance:** The current 60/30/10 weighting may not produce the best rankings for all roles.
4. **Experience extraction:** Simple patterns can miscalculate total experience or miss relevant dates.
5. **Job-description parsing:** Mandatory qualifications, preferred skills, and phrases such as “a plus” may not always be distinguished correctly.
6. **Missing evidence:** A skill not detected in a resume is not necessarily a skill the candidate lacks.
7. **Retrieval and ranking differences:** A relevant candidate may not appear in the top 10, even if the system has indexed that resume.

### Next Evaluation Steps

1. **Verify ground-truth labels.** Review the complete resume and job description for every candidate in each top-10 list. Use `1` for relevant and `0` for not relevant based on consistent criteria.
2. **Check all ranking results.** Inspect the JSON files and determine whether potentially relevant candidates appear below rank 10.
3. **Calculate Recall@10.** This measures how many of all known relevant candidates were retrieved in the top 10. It requires a sufficiently complete set of relevance judgments, not just labels for the current top 10.
4. **Review skill extraction.** Test whether required and preferred skills are parsed correctly, including synonyms and phrases such as “preferred” or “a plus.”
5. **Tune scoring weights.** Compare alternative weighting schemes using the same reviewed evaluation set.
6. **Compare before and after changes.** Recalculate the metrics after each improvement rather than assuming a change has helped.
7. **Consider NDCG@10.** This can evaluate ranking quality when relevance judgments use multiple grades, such as highly relevant, partially relevant, and not relevant.

## Limitations

1. PDF text extraction may fail on scanned or image-only documents; OCR may be required.
2. Experience years are estimated using simple patterns and may not reflect total career experience accurately.
3. The skill vocabulary is finite and should be extended where appropriate.
4. Missing detected skills do not prove that a candidate lacks those skills.
5. A detected skill does not prove proficiency or recent practical experience.
6. The current scoring weights are a baseline and have not been statistically calibrated.
7. Precision@10 depends on the quality and completeness of the relevance labels.
8. The system has not been validated for production hiring decisions.

## Future Improvements

- Improve skill extraction with synonyms and normalization.
- Separate mandatory requirements from preferred qualifications.
- Improve date parsing and total-experience calculations.
- Add human-reviewed relevance labels for a larger evaluation dataset.
- Evaluate Precision@5, Precision@10, Recall@10, and NDCG@10 where appropriate.
- Tune ranking weights on a reviewed validation set.
- Add automated tests for parsing, filtering, and ranking.
- Improve dependency management and reproducibility.
- Add privacy controls, secure storage, bias testing, and human review before any real-world deployment.

## Conclusion

This project demonstrates a local RAG-based resume retrieval and job-matching workflow using semantic embeddings, skill overlap, and experience-based scoring.

The preliminary average Precision@10 is 32% across five job roles. The next priority is to validate the relevance labels and inspect ranking errors. Only after that should the scoring method and retrieval pipeline be adjusted based on measured results.
