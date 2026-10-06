# "Readers also read…": a Goodreads book recommender

SIADS 593 Milestone I, Fall 2026 · Olivia Cain · Sarah Ehlen · David Hook

This repository holds **Book 1 — Data Exploration, Cleaning, and Analysis**, the first of the
project's three notebooks, with the Python modules it calls and the small data files the course
asks us to include. Books 2 (Modeling and Model Evaluation) and 3 (Ranked Fusion Recommender)
will be added as they are finished.

## Interactive explorer

**https://lbdavid98.github.io/milestone_593_cain_ehlen_hook/**

Visualization 1 from the notebook, hosted as a page: every work liked by a training reader, placed
by how many readers liked it and what share of its raters gave it 4–5 stars. Drag the slider to set
a minimum-likes floor, click a genre, type a title, hover for the books behind each cell. The
bottom-right panel compares Goodreads' average rating (to 2017) with Open Library's (2026).

## What is here

| Path | Contents |
|---|---|
| `Book 1 - Data Exploration, Cleaning, and Analysis.ipynb` | the executed notebook (outputs included) |
| `scripts/` | the seven modules the notebook calls: `data_sources`, `book_dimension`, `sample_interactions`, `train_test_split`, `evidence_explorer`, `catalogue_charts`, `open_library` |
| `data/processed/user_splits.pkl` | train / validation / test assignment for the 50,000 sampled readers (seed 593) |
| `data/processed/train_likes_per_work.pkl` | likes per work among training readers; the recommendable floor is applied from it |
| `data/samples/*.first100.csv` | the first 100 records of each output file over 10 MB (`book_dimension`, `interactions_sample_50k`, `open_library_match`) |
| `docs/index.html` | the hosted explorer |
| `requirements.txt` | pinned packages for the Python 3.10 environment the notebook ran in |
| `AI_DISCLOSURE.md` | generative-AI disclosure with the prompts behind each module, as the course policy requires |

## Data

The raw files (about 23 GB) are not in the repository. Section 1 of the notebook lists every
download URL, the path to save each file at, and its expected size, and checks what is present.
The three large processed files are not included either; running the notebook against the raw
downloads produces them (the first full run takes about eight minutes, later runs under two).

Sources: UCSD Goodreads datasets (interactions, book metadata, genres, authors, series;
`https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/`) and the Open Library data dumps of
2026-09-30 (`https://openlibrary.org/developers/dumps`). Open Library's API is described in the
notebook and in `scripts/open_library.py` but is not called by the notebook.

## Running the notebook

```
python3.10 -m pip install -r requirements.txt
python3.10 -m ipykernel install --user --name python310 --display-name "Python 3.10 (milestone1)"
```

Place the raw files as Section 1 describes, open the notebook with that kernel, and run all cells.
Every expensive step writes a `.pkl` under `data/processed/` and loads it on later runs.
