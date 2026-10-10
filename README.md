# "Readers also read…": a Goodreads book recommender

SIADS 593 Milestone I, Fall 2026 · Olivia Cain · Sarah Ehlen · David Hook

A book recommender built from Goodreads reading histories: given a book a reader enjoyed, it
returns five books that readers with similar histories also enjoyed. The work is presented in three
Jupyter notebooks, run in order, with the Python modules they call and the small data files the
course asks us to include.

The project report is `02-scehlen-hookds-ohope_2026fall.pdf` (11 pages: the report on pages 1–10,
references and the statement of work on page 11).

## Interactive explorer

**https://lbdavid98.github.io/milestone_593_cain_ehlen_hook/**

Visualization 1 from Book 1, hosted as a page: every work liked by a training reader, placed by how
many readers liked it and what share of its raters gave it 4–5 stars. Drag the slider to set a
minimum-likes floor, click a genre, type a title, hover for the books behind each cell. The
bottom-right panel compares Goodreads' average rating (to 2017) with Open Library's (2026).

## Recommender app

**https://lbdavid98.github.io/milestone_593_cain_ehlen_hook/app/**

Add books you've read and get five recommendations from Book 3's recommender (the two Jaccard models
fused by rank), with the reason for each pick and a genre comparison against your books; a menu
switches to any single model. The app (`app/`) is a NiceGUI screen in one container on Azure Container
Apps. It scales to zero when idle, so the link above opens a page that shows a loading message while
it wakes up (10–30 seconds), then forwards to the app. `app/build_data.py` bakes the recommender's
output into the image; GitHub Actions rebuilds the image when `app/` changes.

## The notebooks

| Notebook | What it does | Main outputs |
|---|---|---|
| `Book 1 - Data Exploration, Cleaning, and Analysis.ipynb` | Sources and download instructions; UCSD book metadata collapsed from editions to works; a 50,000-reader sample of the interactions at work level; a train/validation/test split by reader; the minimum-evidence decision; the Open Library join through the Goodreads id | `book_dimension.pkl`, `interactions_sample_50k.pkl`, `user_splits.pkl`, `train_likes_per_work.pkl`, `open_library_match.pkl` |
| `Book 2 - Modeling and Model Evaluation.ipynb` | Three recommenders built from one co-occurrence engine on the training readers: a directional score and a Jaccard score on likes, and a Jaccard score on books readers finished; verification of the fast implementations against the reference functions; baselines; evaluation on held-out readers and against Goodreads' own lists | `neighbours_*.pkl` (60 ranked neighbours per work, per model), `book2_test_results.pkl` |
| `Book 3 - Ranked Fusion Recommender.ipynb` | The three models' top-five lists combined with reciprocal rank fusion; tuning on validation readers, where leaving the directional score out wins; the fused recommenders against the single models and baselines on test readers; the project's `recommend(work_id, n=5)`, which fuses the two Jaccard models | `book3_test_results.pkl`, `book3_validation_results.pkl`, `book3_agreement.pkl`, `book3_weight_grid.pkl` |

## Repository layout

| Path | Contents |
|---|---|
| `scripts/` | The modules the notebooks call. Book 1: `data_sources`, `book_dimension`, `sample_interactions`, `train_test_split`, `evidence_explorer`, `catalogue_charts`, `open_library`. Book 2: `cooccurrence`, `likes_models` (with `likes_models_discussion.json`), `finished_together`, `evaluation`, `model_charts`. Book 3: `fusion`, `fusion_charts`, `rank_fusion`. Shared: `chart_theme` |
| `data/processed/` | Outputs under 10 MB: the reader split, training likes per work, and the Book 2 and Book 3 result tables |
| `data/samples/` | The first 100 records of each output over 10 MB: `book_dimension`, `interactions_sample_50k`, `open_library_match`, `similar_books` (Goodreads' "readers also enjoyed" links, the Book 2 benchmark), and the first 100 works of each neighbour table |
| `docs/index.html` | The hosted explorer |
| `requirements.txt` | Pinned packages for the Python 3.10 environment the notebooks ran in |
| `02-scehlen-hookds-ohope_2026fall.pdf` | The project report |

## Submission archive

`02-scehlen-hookds-ohope_2026fall.zip` follows the course layout: the report and this README at the
top level, all code in `src/`, all data in `src/data/`. The notebooks find the modules and data
relative to the folder they are opened from, so they run unchanged from `src/`.

```
02-scehlen-hookds-ohope_2026fall.zip
├── 02-scehlen-hookds-ohope_2026fall.pdf
├── README.md
└── src/
    ├── Book 1 - Data Exploration, Cleaning, and Analysis.ipynb
    ├── Book 2 - Modeling and Model Evaluation.ipynb
    ├── Book 3 - Ranked Fusion Recommender.ipynb
    ├── requirements.txt
    ├── scripts/          the modules listed above
    ├── docs/index.html   the explorer
    └── data/
        ├── processed/    outputs under 10 MB
        └── samples/      first 100 records of each output over 10 MB
```

## Data

The raw files (about 23 GB) are not in the repository. Section 1 of Book 1 lists every download URL,
the path to save each file at and its expected size, and checks what is present. Processed files over
10 MB are not included either; running the notebooks in order produces them. Book 1's first full run
takes about eight minutes and later runs under two; Books 2 and 3 take a few minutes each on the first
run and about a minute afterwards, because every expensive step saves a `.pkl` and loads it when
present.

Sources: the UCSD Goodreads datasets (interactions, book metadata, genres, authors, series;
`https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/`) and the Open Library data dumps of
2026-09-30 (`https://openlibrary.org/developers/dumps`). Open Library's API is described in Book 1 and
in `scripts/open_library.py` but is not called by the notebooks.

## Running the notebooks

```
python3.10 -m pip install -r requirements.txt
python3.10 -m ipykernel install --user --name python310 --display-name "Python 3.10 (milestone1)"
```

Place the raw files as Book 1's Section 1 describes, open the notebooks with that kernel from the
repository root, and run Book 1, Book 2 and Book 3 in order. Every random choice uses seed 593, so a
re-run draws the same readers and assigns them to the same splits.
