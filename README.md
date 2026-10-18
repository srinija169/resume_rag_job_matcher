# Resume RAG & Job Matching Engine

A Retrieval-Augmented Generation (RAG)-based resume retrieval and job-matching system that retrieves relevant resume sections and ranks candidates against job descriptions using semantic similarity and job-related criteria.

## Project Overview

Recruiters often need to review many resumes to identify candidates whose skills and experience align with a job description. This project automates part of that process by indexing resume content, retrieving relevant information, and ranking candidates for different job roles.

The project uses a collection of fictional resumes and job descriptions to demonstrate resume retrieval, candidate matching, and evaluation.

> **Note:** This project is for educational and demonstration purposes only. It should not be used as the sole basis for real hiring decisions.

## Key Features

- Resume ingestion and text chunking
- Semantic embeddings using Sentence Transformers
- Vector search and retrieval using ChromaDB
- Job-description-based candidate matching
- Candidate ranking using matching scores and job-related criteria
- Support for multiple job roles
- Evaluation using Precision@5 and Precision@10
- Notebook-based experiments and evaluation
- Export of matching results for further analysis

## Tech Stack

- **Python** — application logic and data processing
- **Sentence Transformers** — semantic embeddings
- **ChromaDB** — vector storage and similarity search
- **Pandas** — evaluation and tabular results
- **Jupyter Notebook** — experiments and evaluation
- **Git and GitHub** — version control and project hosting

## Project Structure

```text
resume_rag_job_matcher/
├── resumes/
│   └── Fictional resume text files
├── job_descriptions/
│   ├── data_scientist.txt
│   ├── ml_engineer.txt
│   ├── rag_engineer.txt
│   ├── data_engineer.txt
│   └── bi_analyst.txt
├── notebooks/
│   └── experiments.ipynb
├── resume_rag.py
├── job_matcher.py
├── requirements.txt
├── evaluation_results.csv
├── manual_label_review.csv
├── results.json
├── rag_results.json
├── data_engineer_results.json
├── bi_results.json
├── ml_results.json
└── README.md
```

The local ChromaDB index and Python virtual environment are generated locally and should not be committed to GitHub.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/srinija169/resume_rag_job_matcher.git
cd resume_rag_job_matcher
```

### 2. Create a virtual environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## How to Run the Project

### Step 1: Index resumes

Build or reset the vector index from the available resume files:

```bash
python resume_rag.py --reset
```

This processes the resumes, splits their content into chunks, generates embeddings, and stores the vectors in ChromaDB.

### Step 2: Match resumes to a job description

Run the job matcher using the options supported by the script:

```bash
python job_matcher.py --help
```

Use the appropriate command-line arguments shown by the help output to select a job description and retrieve ranked candidates.

The project includes job descriptions for:

- Data Scientist
- Data Engineer
- BI Analyst
- Machine Learning Engineer
- RAG Engineer

### Step 3: Run the experiments notebook

Open the notebook:

```bash
jupyter notebook notebooks/experiments.ipynb
```

Run the notebook cells to explore candidate rankings, review relevance labels, and calculate evaluation metrics.

## System Workflow

1. **Resume ingestion:** Read resume text files from the resumes directory.
2. **Text chunking:** Split resumes into smaller passages for retrieval.
3. **Embedding generation:** Convert text chunks into vector representations.
4. **Vector indexing:** Store embeddings in ChromaDB.
5. **Job description processing:** Use the selected job description as the matching query.
6. **Candidate retrieval and ranking:** Retrieve relevant resume content and rank candidates using the matching logic implemented in the project.
7. **Evaluation:** Compare ranked candidates with manually reviewed relevance labels and calculate Precision@K.

## Evaluation Methodology

The system is evaluated using manually reviewed relevance labels for the ranked candidate results.

**Precision@K** measures the proportion of relevant candidates among the top K ranked candidates:

\[
\text{Precision@K} =
\frac{\text{Relevant candidates in top K}}{K}
\]

For example, if 4 of the top 10 candidates are labeled relevant, Precision@10 is 40%.

## Evaluation Results

The following table records the currently documented evaluation results. These values should be replaced if the latest improved-matching run produces different top-10 relevance counts.

| Job Role | Relevant in Top 10 | Precision@10 |
|---|---:|---:|
| Data Scientist | 3/10 | 30% |
| Data Engineer | 3/10 | 30% |
| BI Analyst | 4/10 | 40% |
| ML Engineer | 4/10 | 40% |
| RAG Engineer | 2/10 | 20% |
| **Overall** | **16/50** | **32%** |

**Average Precision@10: 32%**

The overall figure is calculated from 16 relevant results across 50 ranked candidate positions: 16 ÷ 50 = 32%.

### Precision@5

An earlier evaluation reported the following Precision@5 results:

| Job Role | Relevant in Top 5 | Precision@5 |
|---|---:|---:|
| Data Scientist | 2/5 | 40% |
| Data Engineer | 3/5 | 60% |
| BI Analyst | 4/5 | 80% |
| ML Engineer | 3/5 | 60% |
| RAG Engineer | 2/5 | 40% |
| **Average** | **14/25** | **56%** |

Precision@5 and Precision@10 measure different portions of the ranked lists. They should be calculated from the same final system version when comparing performance.

## Results Analysis

The documented evaluation shows differences in matching performance across job roles.

- **BI Analyst:** Highest documented Precision@10 at 40%.
- **ML Engineer:** Precision@10 of 40%.
- **Data Scientist and Data Engineer:** Precision@10 of 30%.
- **RAG Engineer:** Precision@10 of 20%, indicating room for improvement.

These results suggest that matching quality depends on the role, the resume dataset, the relevance criteria, and the ranking logic. They should be interpreted as results on this evaluation dataset, not as a guarantee of real-world hiring performance.

## Limitations

- The dataset is relatively small and consists of fictional resumes.
- Evaluation results depend on the consistency of relevance labels.
- Semantic similarity does not always imply suitability for a particular role.
- Keyword and skill matching may miss context or transferable skills.
- The system has not been validated as a real-world recruitment decision tool.
- Performance may differ on larger or more diverse resume collections.

## Future Improvements

- Evaluate the improved matching logic on a larger dataset.
- Improve skill extraction and experience matching.
- Tune ranking weights using a held-out evaluation dataset.
- Add more job descriptions and varied resume profiles.
- Compare semantic retrieval against keyword-based baselines.
- Add automated evaluation reports and ranking visualizations.
- Improve retrieval speed and reduce processing latency.
- Add fairness checks and human review for any practical hiring application.

## Conclusion

This project demonstrates how semantic search, vector databases, and job-related ranking criteria can be combined to retrieve and rank resumes against job descriptions.

It also provides an evaluation workflow using Precision@5 and Precision@10 to help identify strengths and areas for improvement.

The project is an educational prototype and should support, rather than replace, human judgment.

## Author

**GitHub:** [srinija169](https://github.com/srinija169)

**Repository:** [Resume RAG & Job Matching Engine](https://github.com/srinija169/resume_rag_job_matcher)
