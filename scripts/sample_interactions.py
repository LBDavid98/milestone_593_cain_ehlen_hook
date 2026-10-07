"""Draw the team's shared sample of Goodreads readers and collapse their shelvings to works.

Book 1 calls these in order:

    users        = sample_user_ids(user_id_map_path)                         # 50,000 reader ids
    raw          = read_sampled_interactions(interactions_csv, users)        # their shelvings
    to_work      = build_edition_to_work_map(books_json_gz, book_id_map_csv) # edition -> work
    interactions = collapse_editions_to_works(raw, to_work)                  # one row per reader and work
    save_interactions(interactions, output_path)

Why a user sample: the full interactions file holds 228.6 million shelvings by
876,145 readers. Sampling whole readers (rather than rows) keeps every connection
inside a reader's history, which is what co-readership recommenders depend on.
50,000 readers is about 5.7% of the population and gives roughly 13 million
shelvings, enough catalogue for evaluation while every model builds in minutes.

Why collapse editions: Goodreads gives every printing its own book id. A reader
who shelved the paperback and another who shelved the hardcover of the same
story would otherwise look like readers of two unrelated books.

Sources (UCSD Goodreads datasets, Mengting Wan and Julian McAuley, 2017-2018):
    goodreads_interactions.csv   user_id, book_id, is_read, rating, is_reviewed
        https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/goodreads_interactions.csv
    user_id_map.csv              user_id_csv -> anonymised Goodreads user hash
        https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/user_id_map.csv
    book_id_map.csv              book_id_csv -> Goodreads book id (an edition)
        https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/book_id_map.csv
    goodreads_books.json.gz      one JSON record per edition, carries work_id
        https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/goodreads_books.json.gz
"""

from __future__ import annotations

import gzip
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- settings
SAMPLE_SIZE = 50_000       # readers in the shared sample
RANDOM_SEED = 593           # fixed so every teammate draws the same readers
CHUNK_SIZE = 5_000_000     # rows per chunk when streaming the 4.3 GB interactions file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DOWNLOADS = Path.home() / "Downloads"

# Narrow integer types: ids fit in int32 and the flags and ratings in int8, which
# keeps each 5-million-row chunk near 60 MB instead of the 200 MB pandas would
# use with its default int64 columns.
INTERACTIONS_DTYPES = {
    "user_id": np.int32, "book_id": np.int32, "is_read": np.int8, "rating": np.int8, "is_reviewed": np.int8,
}


# ----------------------------------------------------------------------------- sampling readers
def count_users(user_id_map_path: str | Path) -> int:
    """Number of readers in the dataset, read from user_id_map.csv (one row per reader)."""
    with open(user_id_map_path, "rb") as f:
        return sum(1 for _ in f) - 1  # minus the header row


def sample_user_ids(user_id_map_path: str | Path | None = None, n_users: int = SAMPLE_SIZE,
                    total_users: int | None = None, seed: int = RANDOM_SEED) -> np.ndarray:
    """Return the sorted ids of the sampled readers.

    Every reader gets a random rank from a seeded permutation; the sample is the
    readers ranked below `n_users`. The same seed therefore always returns the
    same readers, and a bigger sample contains a smaller one.
    """
    if total_users is None:
        if user_id_map_path is None:
            raise ValueError("Give either user_id_map_path or total_users.")
        total_users = count_users(user_id_map_path)
    rank = np.random.default_rng(seed).permutation(total_users)  # rank[user_id] = that reader's random rank
    return np.flatnonzero(rank < n_users).astype(np.int32)        # readers ranked in the top n_users


# ----------------------------------------------------------------------------- reading interactions
def read_sampled_interactions(interactions_csv_path: str | Path, sampled_users: np.ndarray,
                              chunk_size: int = CHUNK_SIZE) -> pd.DataFrame:
    """Stream goodreads_interactions.csv and keep only the sampled readers' rows.

    The file is 228.6 million rows (about 4.3 GB), too large to load at once, so it
    is read in chunks and filtered with a boolean lookup table over user ids.
    Columns returned: user_id, book_id, is_read, rating, is_reviewed, where
    book_id is the CSV's edition index (not yet a Goodreads id).
    """
    # A boolean array indexed by user id is far faster than Series.isin on 228M rows.
    is_sampled = np.zeros(int(sampled_users.max()) + 1, dtype=bool)
    is_sampled[sampled_users] = True
    kept = []
    for chunk in pd.read_csv(interactions_csv_path, dtype=INTERACTIONS_DTYPES, chunksize=chunk_size):
        user = chunk["user_id"].to_numpy()
        # np.minimum guards the lookup against ids larger than the biggest sampled id
        in_sample = (user < len(is_sampled)) & is_sampled[np.minimum(user, len(is_sampled) - 1)]
        kept.append(chunk[in_sample])
    return pd.concat(kept, ignore_index=True)


# ----------------------------------------------------------------------------- editions to works
def build_edition_to_work_map(books_json_gz_path: str | Path, book_id_map_csv_path: str | Path) -> np.ndarray:
    """Array mapping the CSV's edition index (book_id) to a Goodreads work_id.

    Two hops: book_id_map.csv turns the CSV index into a Goodreads book id, and
    goodreads_books.json.gz carries each book's work_id. The books file is 2 GB
    gzipped, so it is scanned line by line with regular expressions instead of
    parsing each record as JSON (about 25 seconds instead of several minutes).
    Editions whose record has an empty work_id map to 0.
    """
    book_id_pattern = re.compile(rb'"book_id": "(\d+)"')
    work_id_pattern = re.compile(rb'"work_id": "(\d*)"')
    work_of_book: dict[int, int] = {}
    with gzip.open(books_json_gz_path, "rb") as f:
        for line in f:
            book = book_id_pattern.search(line)
            work = work_id_pattern.search(line)
            work_of_book[int(book.group(1))] = int(work.group(1)) if work and work.group(1) else 0

    # Second hop as a dense array so the lookup for 13 million rows is one indexing operation.
    id_map = pd.read_csv(book_id_map_csv_path)
    edition_to_work = np.zeros(int(id_map["book_id_csv"].max()) + 1, dtype=np.int64)
    goodreads_ids = id_map["book_id"].to_numpy()
    edition_to_work[id_map["book_id_csv"].to_numpy()] = [work_of_book.get(int(b), 0) for b in goodreads_ids]
    return edition_to_work


def collapse_editions_to_works(interactions: pd.DataFrame, edition_to_work: np.ndarray) -> pd.DataFrame:
    """Replace edition ids with work ids and keep one row per reader and work.

    Rows whose edition has no work id are dropped (a few hundred per million).
    When a reader shelved several editions of one work, the row keeps the highest
    rating and is_read, so a like on any edition counts as a like of the work.
    Returns user_id, work_id, is_read, rating.
    """
    work_id = edition_to_work[interactions["book_id"].to_numpy()]
    has_work = work_id > 0
    print(f"rows without a work id, dropped: {(~has_work).sum():,} of {len(interactions):,}")

    collapsed = (
        pd.DataFrame({
            "user_id": interactions["user_id"].to_numpy()[has_work],
            "work_id": work_id[has_work],
            "is_read": interactions["is_read"].to_numpy()[has_work],
            "rating": interactions["rating"].to_numpy()[has_work],
        })
        .groupby(["user_id", "work_id"], sort=True, as_index=False)
        .max()  # max: a 5-star on one edition and an unrated shelving of another is a 5-star like of the work
    )
    collapsed["work_id"] = collapsed["work_id"].astype(np.int64)
    return collapsed


# ----------------------------------------------------------------------------- saving
def save_interactions(interactions: pd.DataFrame, path: str | Path) -> Path:
    """Write the collapsed sample as a pickle (user_id, work_id, is_read, rating).

    A pickle keeps the dtypes and loads in about a second; the CSV route took ten times
    longer and needed dtype arguments on every read.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    interactions.to_pickle(path)
    return path


def load_or_build_interactions(raw_dir: str | Path, path: str | Path, n_users: int = SAMPLE_SIZE,
                               seed: int = RANDOM_SEED, rebuild: bool = False, log=print) -> pd.DataFrame:
    """Load the saved sample pickle, or build it from the raw files and save it.

    The build is the whole Section 3 pipeline: sample readers, stream their rows out of the
    4.3 GB CSV, map editions to works, collapse duplicate reader-work pairs. About a minute.
    `rebuild=True` ignores the saved file.
    """
    raw_dir, path = Path(raw_dir), Path(path)
    if path.exists() and not rebuild:
        if log:
            log(f"loading {path}")
        return pd.read_pickle(path)
    if log:
        log(f"{path} not found; building the {n_users:,}-reader sample from {raw_dir}")
    users = sample_user_ids(raw_dir / "user_id_map.csv", n_users=n_users, seed=seed)
    raw_rows = read_sampled_interactions(raw_dir / "goodreads_interactions.csv", users)
    edition_to_work = build_edition_to_work_map(raw_dir / "goodreads_books.json.gz", raw_dir / "book_id_map.csv")
    interactions = collapse_editions_to_works(raw_rows, edition_to_work)
    save_interactions(interactions, path)
    return interactions


def write_submission_sample(interactions: pd.DataFrame, path: str | Path, n: int = 100) -> Path:
    """Write the first `n` rows: the course asks for a 100-record sample of any data file over 10 MB."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    interactions.head(n).to_csv(path, index=False)
    return path


def describe_interactions(interactions: pd.DataFrame, like_ratings=(4, 5)) -> pd.Series:
    """Headline counts for the notebook: readers, works, rows, likes, works with a like."""
    liked = interactions[interactions["rating"].isin(like_ratings)]
    return pd.Series({
        "readers": interactions["user_id"].nunique(),
        "works": interactions["work_id"].nunique(),
        "reader-work rows": len(interactions),
        "rows marked read": int(interactions["is_read"].sum()),
        "rows with a rating": int((interactions["rating"] > 0).sum()),
        "likes (4-5 stars)": len(liked),
        "works with at least one like": liked["work_id"].nunique(),
    })


def quality_checks(interactions: pd.DataFrame) -> pd.DataFrame:
    """The three pre-filter checks the notebook reports: missing keys, duplicate pairs, rating range.

    Each row names the check, the count found, and what the notebook does about it. A rating of 0
    is "shelved but not rated" in this file, so it is counted separately rather than flagged as
    missing or out of range.
    """
    key_columns = ["user_id", "book_id", "rating"]
    missing = int(interactions[key_columns].isna().any(axis=1).sum())
    duplicates = int(interactions.duplicated(subset=["user_id", "book_id"]).sum())
    out_of_range = int((~interactions["rating"].between(0, 5)).sum())
    unrated = int((interactions["rating"] == 0).sum())
    return pd.DataFrame([
        {"check": "rows missing user_id, book_id or rating", "found": missing,
         "action": "none needed" if missing == 0 else "dropped"},
        {"check": "duplicate reader-edition pairs", "found": duplicates,
         "action": "none needed" if duplicates == 0 else "collapsed"},
        {"check": "ratings outside 0-5", "found": out_of_range,
         "action": "none needed" if out_of_range == 0 else "investigated"},
        {"check": "rating 0 = shelved but not rated", "found": unrated,
         "action": f"kept ({unrated / len(interactions):.1%} of rows); excluded only from 'likes'"},
    ]).set_index("check")


# ----------------------------------------------------------------------------- file locations
def locate_raw_file(name: str) -> Path:
    """Prefer data/raw/<name>; otherwise copy it there from ~/Downloads so the project is self-contained."""
    target = RAW_DIR / name
    if target.exists():
        return target
    source = DOWNLOADS / name
    if not source.exists():
        raise FileNotFoundError(f"{name} is in neither {RAW_DIR} nor {DOWNLOADS}; download it from UCSD first.")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"copying {source} -> {target}")
    shutil.copy2(source, target)
    return target


def main() -> None:
    started = time.time()
    interactions_csv = locate_raw_file("goodreads_interactions.csv")
    user_id_map = locate_raw_file("user_id_map.csv")
    book_id_map = locate_raw_file("book_id_map.csv")
    books_json = RAW_DIR / "goodreads_books.json.gz"

    users = sample_user_ids(user_id_map)
    assert np.array_equal(users, sample_user_ids(total_users=876_145)), "user count differs from the documented 876,145"
    print(f"[{time.time() - started:5.0f}s] sampled readers: {len(users):,}")

    raw = read_sampled_interactions(interactions_csv, users)
    print(f"[{time.time() - started:5.0f}s] raw shelvings for the sample: {len(raw):,}")

    edition_to_work = build_edition_to_work_map(books_json, book_id_map)
    interactions = collapse_editions_to_works(raw, edition_to_work)
    print(f"[{time.time() - started:5.0f}s] after collapsing editions: {len(interactions):,} reader-work rows")

    out = save_interactions(interactions, PROCESSED_DIR / "interactions_sample_50k.pkl")
    write_submission_sample(interactions, PROCESSED_DIR / "interactions_sample_50k.first100.csv")
    print(f"[{time.time() - started:5.0f}s] saved {out}")
    print(describe_interactions(interactions).to_string())


if __name__ == "__main__":
    main()
