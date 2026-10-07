"""Build the book dimension: one row per Goodreads *work* from the UCSD metadata files.

The notebook calls one function:

    books = load_or_build_book_dimension(raw_dir, "data/processed/book_dimension.pkl")

which loads the saved table when it exists and otherwise rebuilds it from the four
raw files (one to two minutes locally and 2.3 GB of memory). The rebuild is also exposed
step by step so the notebook can narrate each decision:

    names   = load_author_and_series_names(raw_dir)
    editions = read_editions_with_genres(raw_dir, names)      # one row per edition
    editions = fold_english_language_codes(editions)
    books    = collapse_editions_to_works(editions)           # one row per work

Sources (UCSD Goodreads datasets, https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/):
    goodreads_books.json.gz                 2,360,655 editions; everything except genre
    goodreads_book_genres_initial.json.gz   genre weights, same row order as the books file
    goodreads_book_authors.json.gz          author names
    goodreads_book_series.json.gz           series titles

Transformation decisions, carried over from notebook 02 unchanged:
    1. Editions collapse to works. The most-rated edition is canonical and supplies every
       descriptive field (ties broken by lowest book_id); ratings_count and
       text_reviews_count are summed across editions; avg_rating is recomputed as the
       ratings-weighted mean; n_shelves and n_similar are NOT summed because both are
       capped per edition. Genre comes from the canonical edition.
    2. English language codes en, en-US, en-GB, en-CA, en-IN fold into "eng".
       "enm" (Middle English) is left alone.
    3. Dropped: popular_shelves (kept as n_shelves), description (kept as desc_len),
       publisher, format, url, asin, kindle_asin, edition_information, publication_day,
       publication_month, similar_books (built separately as similar_books.npy).
    4. Authors and series stay as lists; genre flattens to ten integer columns.
    5. Numbers arrive as strings and missing as ""; they become 0. Years keep their sign.
    6. Nothing is filtered: every work in the source survives.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

TOTAL_BOOKS = 2_360_655          # editions in goodreads_books.json.gz
ENGLISH_VARIANTS = {"en", "en-US", "en-GB", "en-CA", "en-IN"}

# The genre file labels each edition with up to ten weighted genres. The labels are
# flattened to ten integer columns, 0 meaning the label is absent.
GENRE_LABELS = [
    "children", "comics, graphic", "fantasy, paranormal", "fiction",
    "history, historical fiction, biography", "mystery, thriller, crime",
    "non-fiction", "poetry", "romance", "young-adult",
]
GENRE_COLUMNS = [
    "g_children", "g_comics", "g_fantasy", "g_fiction", "g_history",
    "g_mystery", "g_nonfiction", "g_poetry", "g_romance", "g_youngadult",
]
SUMMED_ACROSS_EDITIONS = ["ratings_count", "text_reviews_count"]

BOOKS_FILE = "goodreads_books.json.gz"
GENRES_FILE = "goodreads_book_genres_initial.json.gz"
AUTHORS_FILE = "goodreads_book_authors.json.gz"
SERIES_FILE = "goodreads_book_series.json.gz"


# ----------------------------------------------------------------------------- reading helpers
def read_json_lines(path):
    """Yield one record per line from a gzip-compressed newline-delimited JSON file."""
    with gzip.open(path, "rt") as source:
        for line in source:
            yield json.loads(line)


def to_whole_number(value):
    """The source stores numbers as strings and missing values as empty strings."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def fit_in_small_integer(value):
    """Counts, clamped to the int16 range."""
    return max(0, min(32767, value))


def fit_year_in_small_integer(value):
    """Years, sign kept so BC dates survive. 0 means missing; there is no year 0."""
    return max(-32767, min(32767, value))


# ----------------------------------------------------------------------------- build steps
def load_author_and_series_names(raw_dir) -> dict:
    """Read the author and series files into id -> name lookups.

    Returns {"authors": {author_id: name}, "series": {series_id: title}}, keyed by the
    string ids the books file uses.
    """
    raw_dir = Path(raw_dir)
    author_name_by_id = {}
    for author in read_json_lines(raw_dir / AUTHORS_FILE):
        author_name_by_id[author["author_id"]] = author["name"]

    series_title_by_id = {}
    for series in read_json_lines(raw_dir / SERIES_FILE):
        series_title_by_id[series["series_id"]] = series["title"]

    return {"authors": author_name_by_id, "series": series_title_by_id}


def read_editions_with_genres(raw_dir, names: dict, total_books: int = TOTAL_BOOKS,
                              log=print) -> pd.DataFrame:
    """Read the books and genre files side by side into one row per edition.

    The two files have the same row count and order, so they are streamed together;
    book ids are compared on every row rather than assumed to line up. Numeric
    fields go straight into preallocated arrays (the file is too large for lists of
    Python objects); text and list fields are collected as Python lists.
    `total_books` sizes the arrays; pass a smaller number when reading a cut-down file.
    """
    raw_dir = Path(raw_dir)

    # Preallocated typed arrays: 2.36 million rows of Python objects would not fit in
    # memory, and the dtypes chosen here (int16 for counts and years, float32 for the
    # rating) are what the saved table carries.
    number_columns = {}
    for column_name in ["book_id", "work_id", "ratings_count", "text_reviews_count", "desc_len"]:
        number_columns[column_name] = np.zeros(total_books, dtype=np.int32)
    for column_name in ["pub_year", "num_pages", "n_similar", "n_shelves"] + GENRE_COLUMNS:
        number_columns[column_name] = np.zeros(total_books, dtype=np.int16)
    number_columns["avg_rating"] = np.zeros(total_books, dtype=np.float32)
    number_columns["is_ebook"] = np.zeros(total_books, dtype=bool)
    number_columns["has_isbn"] = np.zeros(total_books, dtype=bool)

    text_fields = ["title", "title_without_series", "language_code", "country_code",
                   "isbn", "isbn13", "image_url", "link"]
    text_columns = {column_name: [] for column_name in
                    text_fields + ["author_ids", "author_names", "series_ids", "series_names"]}

    books = read_json_lines(raw_dir / BOOKS_FILE)
    genres = read_json_lines(raw_dir / GENRES_FILE)
    row_number = 0
    mismatched_rows = 0

    for book, genre in zip(books, genres):
        if row_number >= total_books:
            raise ValueError(f"the books file has more than {total_books:,} rows; raise total_books")
        if book["book_id"] != genre["book_id"]:
            # the two files are documented as parallel; a mismatch is counted and the
            # edition keeps no genre rather than someone else's (the full files report 0)
            mismatched_rows += 1
            genre = {"genres": {}}

        number_columns["book_id"][row_number] = to_whole_number(book["book_id"])
        number_columns["work_id"][row_number] = to_whole_number(book["work_id"])
        number_columns["ratings_count"][row_number] = to_whole_number(book["ratings_count"])
        number_columns["text_reviews_count"][row_number] = to_whole_number(book["text_reviews_count"])
        number_columns["desc_len"][row_number] = len(book["description"])      # description itself is dropped
        number_columns["pub_year"][row_number] = fit_year_in_small_integer(to_whole_number(book["publication_year"]))
        number_columns["num_pages"][row_number] = fit_in_small_integer(to_whole_number(book["num_pages"]))
        number_columns["n_similar"][row_number] = fit_in_small_integer(len(book["similar_books"]))  # list -> count
        number_columns["n_shelves"][row_number] = fit_in_small_integer(len(book["popular_shelves"]))  # list -> count
        number_columns["avg_rating"][row_number] = float(book["average_rating"] or 0)
        number_columns["is_ebook"][row_number] = book["is_ebook"] == "true"
        number_columns["has_isbn"][row_number] = bool(book["isbn"]) or bool(book["isbn13"])

        for label, column_name in zip(GENRE_LABELS, GENRE_COLUMNS):
            weight = genre["genres"].get(label)
            if weight:
                number_columns[column_name][row_number] = fit_in_small_integer(weight)

        author_ids, author_names = [], []
        for credited_author in book["authors"]:
            author_id = credited_author["author_id"]
            author_ids.append(to_whole_number(author_id))
            author_names.append(names["authors"].get(author_id, ""))
        series_ids, series_names = [], []
        for series_id in book["series"]:
            series_ids.append(to_whole_number(series_id))
            series_names.append(names["series"].get(series_id, ""))

        text_columns["author_ids"].append(author_ids)
        text_columns["author_names"].append(author_names)
        text_columns["series_ids"].append(series_ids)
        text_columns["series_names"].append(series_names)
        for field in text_fields:
            text_columns[field].append(book[field])

        row_number += 1
        if log and row_number % 500_000 == 0:
            log(f"{row_number:,} books read")

    if log:
        log(f"finished: {row_number:,} books; rows where the genre file did not line up: {mismatched_rows:,}")

    # a cut-down file leaves unused rows at the end of the arrays; trim them
    editions = pd.DataFrame({name: values[:row_number] for name, values in number_columns.items()})
    for column_name in ["title", "title_without_series", "author_ids", "author_names",
                        "series_ids", "series_names", "isbn", "isbn13", "image_url", "link",
                        "language_code", "country_code"]:
        editions[column_name] = text_columns[column_name]
    return editions


def fold_english_language_codes(editions: pd.DataFrame) -> pd.DataFrame:
    """Map en, en-US, en-GB, en-CA and en-IN onto "eng"; "enm" (Middle English) is left alone."""
    # Filtering on "eng" alone would have dropped 157,649 English editions tagged with a
    # regional variant; folding them first keeps the language filter honest downstream.
    editions = editions.copy()
    editions["language_code"] = editions["language_code"].replace(
        {variant: "eng" for variant in ENGLISH_VARIANTS})
    return editions


def collapse_editions_to_works(editions: pd.DataFrame) -> pd.DataFrame:
    """One row per work: the most-rated edition is canonical, counts are summed, the mean re-weighted."""
    # Canonical edition: within each work, sort by ratings (most first) and break ties on
    # the lowest book_id, then keep the first row. Every descriptive field (title, pages,
    # year, genre, ISBN, language...) comes from this one real edition rather than from a
    # blend of editions that never existed. n_shelves and n_similar come along as they are:
    # both are capped per edition (100 shelves, 18 similar books), so summing them across
    # editions would produce a number with no meaning.
    ordered = editions.sort_values(["work_id", "ratings_count", "book_id"], ascending=[True, False, True])
    canonical = ordered.groupby("work_id", sort=True).head(1).set_index("work_id")

    # Counts that are true totals (ratings, text reviews) ARE summed across editions, so
    # a story printed fifty times keeps all of its readers. The average rating is then
    # rebuilt as the ratings-weighted mean of the editions' averages, which is exactly the
    # mean Goodreads would show if the editions had been one listing.
    grouped = editions.groupby("work_id", sort=True)
    summed = grouped[SUMMED_ACROSS_EDITIONS].sum()
    weighted_rating = editions["avg_rating"].astype(np.float64) * editions["ratings_count"]
    weighted_total = weighted_rating.groupby(editions["work_id"], sort=True).sum()

    books = canonical.drop(columns=SUMMED_ACROSS_EDITIONS).copy()
    books = books.rename(columns={"book_id": "canonical_book_id"})
    for column_name in SUMMED_ACROSS_EDITIONS:
        books[column_name] = summed[column_name]

    # works with no ratings at all get 0.0 rather than a division by zero
    total_ratings = books["ratings_count"].to_numpy()
    books["avg_rating"] = np.where(
        total_ratings > 0, weighted_total.to_numpy() / np.maximum(total_ratings, 1), 0.0
    ).astype(np.float32)

    books = books.reset_index()
    # a few hundred distinct codes over 1.5 million rows: categoricals cut the memory to a fraction
    for column_name in ["language_code", "country_code"]:
        books[column_name] = pd.Categorical(books[column_name])

    assert books["work_id"].is_unique
    assert books["ratings_count"].sum() == editions["ratings_count"].sum(), "ratings must be preserved exactly"
    return books


def build_book_dimension(raw_dir, total_books: int = TOTAL_BOOKS, log=print) -> pd.DataFrame:
    """Run the whole build: names, editions, language fold, collapse to works."""
    names = load_author_and_series_names(raw_dir)
    if log:
        log(f"author names {len(names['authors']):,}; series titles {len(names['series']):,}")
    editions = read_editions_with_genres(raw_dir, names, total_books=total_books, log=log)
    editions = fold_english_language_codes(editions)
    books = collapse_editions_to_works(editions)
    if log:
        log(f"editions in {len(editions):,}; works out {len(books):,} "
            f"({1 - len(books) / len(editions):.1%} collapsed); columns {books.shape[1]}")
    return books


def save_book_dimension(books: pd.DataFrame, path) -> None:
    """Pickle the table (pickle keeps the list columns and categoricals intact)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    books.to_pickle(path)


def load_or_build_book_dimension(raw_dir, path, rebuild: bool = False, log=print) -> pd.DataFrame:
    """Load the saved book dimension, or build and save it when the file is missing.

    rebuild=True ignores the saved file and runs the build again (one to two minutes on a laptop,
    eight in Colab); the notebook exposes that as a flag so a grader can reproduce the table from
    the raw files.
    """
    path = Path(path)
    if path.exists() and not rebuild:
        if log:
            log(f"loading {path}")
        return pd.read_pickle(path)
    if log:
        log(f"{path} not found; building from {raw_dir} (one to two minutes locally)")
    books = build_book_dimension(raw_dir, log=log)
    save_book_dimension(books, path)
    return books


# ----------------------------------------------------------------------------- notebook helpers
def read_first_record(books_json_gz_path) -> dict:
    """The first raw edition record, for showing what the source looks like before any cleaning."""
    for record in read_json_lines(books_json_gz_path):
        return record
    raise ValueError(f"{books_json_gz_path} is empty")


USED_COLUMNS = {   # every column the three notebooks read, with what each is for
    "work_id": "key; joins to interactions, similar_books and the Open Library match (Books 1-3)",
    "canonical_book_id": "Goodreads edition id of the most-rated edition; the Open Library join key (Book 1)",
    "title": "shown in explorer tooltips and recommendation lists (Books 1-3)",
    "author_names": "list; first author shown with each title (Books 1-3)",
    "author_ids": "list; the same-author filter on recommendations (Book 2)",
    "series_ids": "list; the same-series filter on recommendations (Book 2)",
    "series_names": "list; shown when a recommendation is in a series (Books 2-3)",
    "pub_year": "canonical edition's year, signed (BC negative); decade view, recency checks (Books 1-2)",
    "avg_rating": "ratings-weighted mean across editions; compared with Open Library ratings (Book 1)",
    "ratings_count": "summed across editions; popularity bands, concentration curve, collapse check (Books 1-2)",
}


def describe_schema(books: pd.DataFrame, used: dict = USED_COLUMNS) -> pd.DataFrame:
    """The columns the project reads, one row each, then the ten genre columns as one row and
    the remaining columns as one row. Replaces a 34-row dtype dump."""
    example = books.loc[books["ratings_count"].idxmax()]      # the most-rated work makes a readable example
    rows = [{"column": c, "dtype": str(books[c].dtype), "example": str(example[c])[:40], "used for": why}
            for c, why in used.items() if c in books.columns]
    genres = [c for c in books.columns if c.startswith("g_")]
    rows.append({"column": f"{genres[0]} ... {genres[-1]} ({len(genres)} columns)",
                 "dtype": str(books[genres[0]].dtype),
                 "example": "0 = label absent", "used for": "top genre in the explorer; genre similarity (Books 1-2)"})
    rest = [c for c in books.columns if c not in used and c not in genres]
    rows.append({"column": f"{len(rest)} other columns", "dtype": "", "example": "",
                 "used for": "carried but not read by the notebooks: " + ", ".join(rest)})
    return pd.DataFrame(rows).set_index("column")


def collapse_summary(books: pd.DataFrame, editions_in_source: int = TOTAL_BOOKS,
                     ratings_in_source: int = 958_938_616) -> pd.DataFrame:
    """Before/after accounting for the edition-to-work collapse.

    The source figures are the counts notebook 02 measured when it read every edition
    (2,360,655 rows, 958,938,616 ratings). The check that matters is the last row: the
    ratings total after the collapse must equal the total in the source, which shows the
    collapse relabelled rows rather than dropping evidence.
    """
    ratings_after = int(books["ratings_count"].sum())
    memory_gb = books.memory_usage(deep=True).sum() / 1e9
    return pd.DataFrame([
        {"measure": "rows", "source (editions)": f"{editions_in_source:,}",
         "book dimension (works)": f"{len(books):,}", "note": f"{1 - len(books) / editions_in_source:.1%} fewer rows"},
        {"measure": "columns", "source (editions)": "29 (+10 genre weights)",
         "book dimension (works)": str(books.shape[1]), "note": "7 dropped, lists kept, genres flattened"},
        {"measure": "ratings counted", "source (editions)": f"{ratings_in_source:,}",
         "book dimension (works)": f"{ratings_after:,}",
         "note": "preserved exactly" if ratings_after == ratings_in_source else "MISMATCH: investigate"},
        {"measure": "in memory", "source (editions)": "2.34 GB", "book dimension (works)": f"{memory_gb:.2f} GB",
         "note": ""},
    ]).set_index("measure")
