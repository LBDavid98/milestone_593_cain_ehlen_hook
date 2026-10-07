"""Where every raw file comes from, and a check that they are in place.

Book 1 does not ship the raw data (about 18 GB). This module is the single list of
download URLs, target paths and expected sizes, so the notebook can print exact
instructions and confirm what is present before any step runs.

    inventory(project_root)   one row per raw file: present?, size on disk, expected size
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

UCSD_BASE = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/"
UCSD_PAGE = "https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html"
OPEN_LIBRARY_DUMPS = "https://openlibrary.org/developers/dumps"

# (relative path inside the project, download URL, approximate size in bytes, what it is)
# Open Library's "latest" links redirect to a dated file; we used the 2026-09-30 dump and
# keep the date in the file name so a later dump does not silently change the join.
RAW_FILES = [
    ("data/raw/goodreads_interactions.csv", UCSD_BASE + "goodreads_interactions.csv", 4_318_621_741,
     "228,648,342 reader-edition interactions: user_id, book_id, is_read, rating, is_reviewed"),
    ("data/raw/book_id_map.csv", UCSD_BASE + "book_id_map.csv", 37_846_957,
     "edition index used in the interactions CSV -> Goodreads book id"),
    ("data/raw/user_id_map.csv", UCSD_BASE + "user_id_map.csv", 34_934_710,
     "reader index used in the interactions CSV -> anonymised Goodreads user hash"),
    ("data/raw/goodreads_books.json.gz", UCSD_BASE + "goodreads_books.json.gz", 2_083_197_934,
     "2,360,655 editions with work_id, title, authors, ratings, ISBNs, publication data"),
    ("data/raw/goodreads_book_genres_initial.json.gz", UCSD_BASE + "goodreads_book_genres_initial.json.gz",
     24_239_233, "ten genre weights per edition"),
    ("data/raw/goodreads_book_authors.json.gz", UCSD_BASE + "goodreads_book_authors.json.gz", 17_988_861,
     "author id -> name"),
    ("data/raw/goodreads_book_series.json.gz", UCSD_BASE + "goodreads_book_series.json.gz", 28_350_926,
     "series id -> title"),
    ("data/raw/openlibrary/ol_dump_editions_2026-09-30.txt.gz",
     "https://openlibrary.org/data/ol_dump_editions_latest.txt.gz", 12_617_475_043,
     "every Open Library edition; carries the Goodreads identifier we join on"),
    ("data/raw/openlibrary/ol_dump_works_2026-09-30.txt.gz",
     "https://openlibrary.org/data/ol_dump_works_latest.txt.gz", 4_073_320_823,
     "every Open Library work; subjects live here"),
    ("data/raw/openlibrary/ol_dump_ratings_2026-09-30.txt.gz",
     "https://openlibrary.org/data/ol_dump_ratings_latest.txt.gz", 9_323_906,
     "one line per star rating given on Open Library"),
]


def inventory(project_root: str | Path) -> pd.DataFrame:
    """One row per raw file: whether it is on disk and whether its size matches the download.

    A size mismatch usually means an interrupted download; the gzip files read without
    error up to the cut, so the size check is the reliable test.
    """
    root = Path(project_root)
    rows = []
    for relative, url, expected, _ in RAW_FILES:
        path = root / relative
        size = path.stat().st_size if path.exists() else 0
        rows.append({
            "file": relative,
            "present": path.exists(),
            "size on disk (GB)": round(size / 1e9, 3),
            "expected (GB)": round(expected / 1e9, 3),
            "complete": size == expected,
            "url": url,
        })
    return pd.DataFrame(rows).set_index("file")


def download_table() -> pd.DataFrame:
    """The download instructions as a table: save each URL to the given path."""
    return pd.DataFrame(
        [{"save as": rel, "download from": url, "size (GB)": round(size / 1e9, 2), "contents": what}
         for rel, url, size, what in RAW_FILES]
    ).set_index("save as")
