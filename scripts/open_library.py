"""Open Library: the live lookup API (Sarah's functions) and readers for the monthly dumps.

Open Library (https://openlibrary.org) is the project's second data source. It is
reached two ways, and the notebook uses both:

1. Live lookup, one book at a time (Section 1, Sarah Ehlen's code, unchanged).
   This is the path the dashboard takes when a reader types a title, author or ISBN:

       record = get_book_record(title="The Hobbit", author="J.R.R. Tolkien")

   Endpoints called, all JSON:
       https://openlibrary.org/isbn/{isbn}.json           edition by ISBN
       https://openlibrary.org/search.json?q=...          title / author search
       https://openlibrary.org/works/{id}.json            the work behind an edition
       https://openlibrary.org/works/{id}/ratings.json    community star ratings
       https://openlibrary.org/tags/{id}.json             name of a genre tag
   Rate limit (https://openlibrary.org/developers/api): 1 request per second for
   unidentified requests, 3 per second when the User-Agent header names the
   application and a contact email, which Sarah's code does. The same page asks
   that bulk metadata come from the monthly dumps rather than the API, which is
   why the catalogue-wide join below uses the dumps.

2. The monthly data dumps (Section 2, ours), for joining the whole Goodreads
   catalogue to Open Library offline:

       matches  = scan_editions_for_goodreads_ids(editions_dump, books["canonical_book_id"])
       subjects = read_work_subjects(works_dump, matches["ol_work_key"])
       ratings  = read_ratings_dump(ratings_dump)
       books    = attach_open_library_to_books(books, matches, subjects, ratings)

   Dump files (https://openlibrary.org/developers/dumps), dated 2026-09-30, saved
   under data/raw/openlibrary/:
       ol_dump_editions_2026-09-30.txt.gz   12.6 GB   one edition per line
       ol_dump_works_2026-09-30.txt.gz       4.1 GB   one work per line
       ol_dump_ratings_2026-09-30.txt.gz     9.3 MB   one star rating per line
   The "latest" links resolve to these files:
       https://openlibrary.org/data/ol_dump_editions_latest.txt.gz
       https://openlibrary.org/data/ol_dump_works_latest.txt.gz
       https://openlibrary.org/data/ol_dump_ratings_latest.txt.gz
   Editions and works are tab-separated with five columns: record type, key,
   revision, last-modified timestamp, and the full record as JSON. The ratings
   dump has four columns: work key, edition key (or \\N), rating 1-5, date.

   The join key is the Goodreads book id. Open Library editions carry it under
   identifiers.goodreads, and the UCSD book dimension keeps each work's most-rated
   edition as canonical_book_id, so the two sources meet on an id both publish.

Authorship: Section 1 is Sarah Ehlen's code, copied verbatim from her notebook
data_manipulation_stage.ipynb. Section 2 and this docstring were authored by
generative AI (Claude Code, Anthropic, 2026-10-06) under David Hook's direction and
reviewed by him, as the SIADS 593 generative-AI policy requires. The prompts that
produced them are recorded in AI_DISCLOSURE.md at the project root.
"""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ============================================================================= Section 1
# Sarah Ehlen's live-lookup functions, copied verbatim from
# final-staging/data_manipulation_stage.ipynb (cells 16f4744e, 34611b6f, fbb18c74).
# Nothing inside this section has been edited; see the notebook's Markdown for the
# known limits of the search path (the search endpoint does not return the fields
# completeness_score weighs, so title searches come back without ISBN, pages or genre).
# =============================================================================


def completeness_score(book):
    """
    Score an OpenLibrary search result based on how much useful
    metadata it contains.
    """
    fields = {
        "title": 1,
        "authors": 2,
        "identifiers": 2,
        "isbn": 5,
        "number_of_pages": 4,
        "genres": 4,
        "first_publish_year": 2,
        "language": 1,
    }

    score = 0

    for field, weight in fields.items():
        value = book.get(field)

        if value is None:
            continue

        if isinstance(value, list):
            # Give credit for non-empty lists
            if len(value) > 0:
                score += weight
        elif value != "":
            score += weight

    # if book.get('language') != ['eng']: # Non-English books are less likely to be relevant to our analysis. Score = 0
    #     score = -1

    return score


def get_tag_name(tag_key, headers=None):
    if tag_key is None:
        return None

    # Handle dictionaries
    if isinstance(tag_key, dict):
        tag_key = tag_key.get("key") or tag_key.get("name")

    if not isinstance(tag_key, str):
        return None

    tag_key = tag_key.strip()

    # Extract the tag ID from paths such as /tags/OL169T
    if "/" in tag_key:
        tag_id = tag_key.rstrip("/").split("/")[-1]
    else:
        tag_id = tag_key

    if not tag_id:
        return None

    url = f"https://openlibrary.org/tags/{tag_id}.json"

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=15,
            allow_redirects=False
        )

        # If OpenLibrary redirects the request, don't follow
        # the problematic redirect automatically.
        if response.status_code in (301, 302, 303, 307, 308):
            return None

        if response.status_code != 200:
            return None

        payload = response.json()

        return payload.get("name") or tag_id

    except (requests.RequestException, UnicodeDecodeError, ValueError):
        return None


def get_book_ratings(work_key, headers=None):
    if headers is None:
        headers = {"User-Agent": "MADSMilestone1/1.0 (scehlen@umich.edu)"}

    url = f"https://openlibrary.org{work_key}/ratings.json"
    response = requests.get(url, headers=headers, timeout=15)

    if response.status_code != 200:
        return None

    return response.json()


def get_book_record(title=None, author=None, isbn=None):
    """
    Fetch a book from OpenLibrary using ISBN, title, or title + author.
    Resolves to the most recent edition, fetches community reviews via the
    Works API, and returns a one-row pandas DataFrame.
    """
    if isbn is None and title is None:
        raise ValueError("Provide at least one of: isbn, title, or title + author.")

    headers = {"User-Agent": "MADSMilestone1/1.0 (scehlen@umich.edu)"}
    data = {}
    work_key = None
    works_json = {}
    ratings_data = None

    # 1. Fetching base metadata
    if isbn is not None:
        clean_isbn = str(isbn).strip().replace("-", "")
        url = f"https://openlibrary.org/isbn/{clean_isbn}.json"
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            raise ValueError(f"OpenLibrary did not return a result for ISBN {clean_isbn}.")
        data = response.json()

        if "works" in data and isinstance(data["works"], list) and len(data["works"]) > 0:
            work_key = data["works"][0].get("key")

    else:
        query_parts = []
        if title:
            query_parts.append(str(title).strip())
        if author:
            query_parts.append(f"author:{str(author).strip()}")
        query = " ".join(part for part in query_parts if part)

        if not query:
            raise ValueError("Provide a title or author for the search.")

        search_url = "https://openlibrary.org/search.json"
        search_response = requests.get(
            search_url,
            params={"q": query},
            headers=headers,
            timeout=15
        )

        if search_response.status_code != 200:
            raise ValueError(f"Search failed for: {query}")

        docs = search_response.json().get("docs", [])
        if not docs:
            raise ValueError(f"No books matched your search: {query}")

        data = max(docs, key=completeness_score)
        work_key = data.get("key")

    # 2. Call the Works API and ratings API
    if work_key:
        if not work_key.startswith("/works/"):
            work_key = f"/works/{work_key}"

        works_url = f"https://openlibrary.org{work_key}.json"
        works_response = requests.get(works_url, headers=headers, timeout=15)

        if works_response.status_code == 200:
            works_json = works_response.json()
            ratings_data = get_book_ratings(work_key)

    # Authors parsing
    raw_authors = data.get("authors", []) or data.get("author_name", [])
    if isinstance(raw_authors, list):
        author_names = []
        for author_obj in raw_authors:
            if isinstance(author_obj, dict):
                if "name" in author_obj:
                    author_names.append(author_obj["name"])
                elif "key" in author_obj:
                    author_names.append(author_obj["key"].split("/")[-1].replace("_", " "))
            elif isinstance(author_obj, str):
                author_names.append(author_obj)
        authors = ", ".join(author_names)
    else:
        authors = str(raw_authors)

    # ISBN parsing
    isbn_value = (
        data.get("isbn") or
        (data.get("isbn_10", [None])[0] if isinstance(data.get("isbn_10"), list) else data.get("isbn_10")) or
        (data.get("isbn_13", [None])[0] if isinstance(data.get("isbn_13"), list) else data.get("isbn_13")) or
        None
    )
    if isinstance(isbn_value, list):
        isbn_value = isbn_value[0] if isbn_value else None

    # Genre tag mapping
    genres = works_json.get("genres", [])
    if isinstance(genres, str):
        genres = [genres]

    genre_names = []

    for tag in genres:
        try:
            tag_name = get_tag_name(tag)

            if tag_name:
                genre_names.append(tag_name)

        except (requests.RequestException, UnicodeDecodeError, ValueError):
            continue

    genre_names = list(dict.fromkeys(genre_names))

    # Extract rating summary and counts
    summary = ratings_data.get("summary", {}) if isinstance(ratings_data, dict) else {}
    counts = ratings_data.get("counts", {}) if isinstance(ratings_data, dict) else {}

    # Extract goodreads identifier if available
    goodreads_identifier = data.get("identifiers", {}).get("goodreads", [None])[0] if isinstance(data.get("identifiers", {}).get("goodreads"), list) else data.get("identifiers", {}).get("goodreads")

    row = {
        "isbn": isbn_value,
        "title": data.get("title") or data.get("title_suggest"),
        "authors": authors,
        "first_publish_year": data.get("first_publish_year"),
        "num_of_pages": data.get("number_of_pages"),
        "genre": ", ".join(genre_names),
        "completeness_score": completeness_score(data),
        "average_stars": summary.get("average"),
        "review_count": summary.get("count"),
        "star_counts": counts if counts else None,
        "goodreads_identifier": goodreads_identifier,
        "works_id": work_key
    }

    return pd.DataFrame([row])


# ============================================================================= Section 2
# Readers for the monthly dumps. Each dump is streamed line by line through gzip so
# memory stays flat; the editions dump alone is 12.6 GB compressed.
# =============================================================================

# Column positions in the editions and works dumps (tab-separated).
DUMP_TYPE, DUMP_KEY, DUMP_REVISION, DUMP_MODIFIED, DUMP_JSON = range(5)


def _dump_lines(path):
    """Yield raw bytes lines from a gzip dump. Bytes, not str: the substring pre-filters
    below work on bytes, and decoding only the lines we keep is far cheaper."""
    with gzip.open(path, "rb") as source:
        for line in source:
            yield line


def _first(values):
    """First element of a list-valued field, or None when the field is missing or empty."""
    if isinstance(values, list) and values:
        return values[0]
    if isinstance(values, str) and values:
        return values
    return None


def scan_editions_for_goodreads_ids(editions_dump_path, wanted_goodreads_ids, progress_every: int = 5_000_000,
                                    max_lines: int | None = None, log=print) -> pd.DataFrame:
    """Find the Open Library editions that carry one of our Goodreads book ids.

    Streams the editions dump once. Only editions whose JSON mentions "goodreads"
    are parsed: the substring test on the raw bytes is roughly a hundred times
    cheaper than json.loads, and most editions have no Goodreads identifier at all.

    wanted_goodreads_ids: the Goodreads book ids to look for (ints or strings), e.g.
                          books["canonical_book_id"].
    max_lines:            stop after this many lines (for tests on a partial dump).

    Returns one row per matched (goodreads_id, edition): goodreads_id (int),
    ol_edition_key, ol_work_key, isbn_13, isbn_10, title. A Goodreads id can match
    more than one edition; attach_open_library_to_books() keeps the first.
    """
    wanted = {str(int(x)) for x in pd.Series(list(wanted_goodreads_ids)).dropna().astype(int)}
    needle = b'"goodreads"'   # the identifiers dict key, as it appears in the JSON column
    rows = []
    started = time.time()

    for line_number, line in enumerate(_dump_lines(editions_dump_path), start=1):
        if max_lines and line_number > max_lines:
            break
        if log and line_number % progress_every == 0:
            log(f"{line_number:,} editions scanned, {len(rows):,} matches, {time.time() - started:,.0f}s")
        if needle not in line:
            continue
        record = json.loads(line.split(b"\t", 4)[DUMP_JSON])
        goodreads_ids = record.get("identifiers", {}).get("goodreads", [])
        hits = [g for g in goodreads_ids if g in wanted]
        if not hits:
            continue
        works = record.get("works", [])
        work_key = works[0].get("key") if works else None
        for goodreads_id in hits:
            rows.append({
                "goodreads_id": int(goodreads_id),
                "ol_edition_key": record.get("key"),
                "ol_work_key": work_key,
                "isbn_13": _first(record.get("isbn_13")),
                "isbn_10": _first(record.get("isbn_10")),
                "title": record.get("title"),
            })

    if log:
        log(f"done: {line_number:,} editions scanned, {len(rows):,} matches in {time.time() - started:,.0f}s")
    columns = ["goodreads_id", "ol_edition_key", "ol_work_key", "isbn_13", "isbn_10", "title"]
    return pd.DataFrame(rows, columns=columns)


def read_work_subjects(works_dump_path, wanted_work_keys, progress_every: int = 5_000_000,
                       max_lines: int | None = None, log=print) -> pd.DataFrame:
    """Pull title, subjects and first publish date for the wanted works from the works dump.

    The key sits in the dump's second column, so each line is accepted or rejected
    on a split of the first few bytes without parsing JSON. Subjects are Open
    Library's free-text tags ("Fantasy", "Hobbits", ...), far finer than the ten
    UCSD genre columns, which is what the join is for.

    Returns work_key, title, subjects (list), subject_count, first_publish_date.
    """
    wanted = set(pd.Series(list(wanted_work_keys)).dropna().astype(str))
    rows = []
    started = time.time()

    for line_number, line in enumerate(_dump_lines(works_dump_path), start=1):
        if max_lines and line_number > max_lines:
            break
        if log and line_number % progress_every == 0:
            log(f"{line_number:,} works scanned, {len(rows):,} found, {time.time() - started:,.0f}s")
        parts = line.split(b"\t", 2)      # type, key, rest; the key decides whether we parse
        if len(parts) < 3 or parts[DUMP_KEY].decode() not in wanted:
            continue
        record = json.loads(line.split(b"\t", 4)[DUMP_JSON])
        subjects = [s for s in record.get("subjects", []) if isinstance(s, str)]
        rows.append({
            "work_key": record.get("key"),
            "title": record.get("title"),
            "subjects": subjects,
            "subject_count": len(subjects),
            "first_publish_date": record.get("first_publish_date"),
        })

    if log:
        log(f"done: {line_number:,} works scanned, {len(rows):,} found in {time.time() - started:,.0f}s")
    columns = ["work_key", "title", "subjects", "subject_count", "first_publish_date"]
    return pd.DataFrame(rows, columns=columns)


def read_ratings_dump(ratings_dump_path) -> pd.DataFrame:
    """Aggregate the ratings dump to one row per work: how many stars were given, and the mean.

    The dump is small (one line per individual rating) so it is read whole. Ratings
    given to a specific edition still carry the work key, so everything rolls up to
    the work. Returns work_key, rating_count, rating_mean.
    """
    ratings = pd.read_csv(ratings_dump_path, sep="\t", header=None, na_values=["\\N"],
                          names=["work_key", "edition_key", "rating", "date"],
                          dtype={"work_key": str, "edition_key": str, "rating": "Int64", "date": str})
    ratings = ratings.dropna(subset=["work_key", "rating"])
    per_work = ratings.groupby("work_key")["rating"].agg(rating_count="size", rating_mean="mean").reset_index()
    per_work["rating_mean"] = per_work["rating_mean"].astype(float).round(3)
    return per_work


def attach_open_library_to_books(books: pd.DataFrame, matches: pd.DataFrame, subjects: pd.DataFrame,
                                 ratings: pd.DataFrame) -> pd.DataFrame:
    """Join the dump results onto the book dimension, one row per work as before.

    books:    the UCSD book dimension (needs work_id and canonical_book_id).
    matches:  from scan_editions_for_goodreads_ids(); keyed by the canonical Goodreads id.
    subjects: from read_work_subjects(); keyed by Open Library work key.
    ratings:  from read_ratings_dump(); keyed by Open Library work key.

    Adds ol_work_key, ol_subject_count, ol_subjects (list; empty when unmatched),
    ol_rating_count, ol_rating_mean, and matched_on ("goodreads_id" or "none").
    Works never lose a row by being unmatched: the join is left, and a missing
    match shows up as 0 subjects, 0 ratings and matched_on == "none".
    """
    # one edition per Goodreads id: the dump lists editions in key order, keep the first seen
    first_match = matches.drop_duplicates("goodreads_id").set_index("goodreads_id")["ol_work_key"]

    out = books.copy()
    out["ol_work_key"] = out["canonical_book_id"].map(first_match)
    out["matched_on"] = np.where(out["ol_work_key"].notna(), "goodreads_id", "none")

    by_work = subjects.drop_duplicates("work_key").set_index("work_key")
    out["ol_subject_count"] = out["ol_work_key"].map(by_work["subject_count"]).fillna(0).astype(int)
    out["ol_subjects"] = out["ol_work_key"].map(by_work["subjects"])
    out["ol_subjects"] = out["ol_subjects"].apply(lambda s: s if isinstance(s, list) else [])

    by_rating = ratings.drop_duplicates("work_key").set_index("work_key")
    out["ol_rating_count"] = out["ol_work_key"].map(by_rating["rating_count"]).fillna(0).astype(int)
    out["ol_rating_mean"] = out["ol_work_key"].map(by_rating["rating_mean"])
    return out


def dump_is_complete(path, expected_bytes: int) -> bool:
    """True when the downloaded dump has its full size. A partial gzip reads fine up to the
    point it was cut, so the size check is what tells a finished download from one in flight."""
    return Path(path).exists() and Path(path).stat().st_size == expected_bytes


EXPECTED_DUMP_BYTES = {   # content-length reported by archive.org for the 2026-09-30 dumps
    "ol_dump_editions_2026-09-30.txt.gz": 12_617_475_043,
    "ol_dump_works_2026-09-30.txt.gz": 4_073_320_823,
    "ol_dump_ratings_2026-09-30.txt.gz": 9_323_906,
}


# ----------------------------------------------------------------------------- notebook helpers
def load_or_build_open_library_match(books: pd.DataFrame, dump_dir, cache_path, rescan: bool = False,
                                     max_lines: int | None = None, log=print) -> pd.DataFrame:
    """Join the book dimension to Open Library through the dumps, once, and cache the result.

    Steps: (1) scan the editions dump for editions carrying one of our canonical Goodreads
    book ids and note their Open Library work; (2) read those works' subjects from the works
    dump; (3) aggregate the ratings dump per work. The result is one row per matched UCSD
    work: work_id, canonical_book_id, ol_work_key, ol_edition_key, isbn_13, ol_title,
    ol_subject_count, ol_subjects (list), ol_first_publish_date, ol_rating_count,
    ol_rating_mean. The result is cached as a pickle so the lists survive as lists.

    The editions scan reads 12.6 GB and takes about three minutes; the result is
    cached because it never changes between runs; `rescan=True` forces all three steps to run
    again. `max_lines` limits the two
    big scans for tests.
    """
    cache_path = Path(cache_path)
    dump_dir = Path(dump_dir)
    if cache_path.exists() and not rescan:
        if log:
            log(f"loading {cache_path}")
        return pd.read_pickle(cache_path)

    editions_dump = dump_dir / "ol_dump_editions_2026-09-30.txt.gz"
    works_dump = dump_dir / "ol_dump_works_2026-09-30.txt.gz"
    ratings_dump = dump_dir / "ol_dump_ratings_2026-09-30.txt.gz"
    for dump in (editions_dump, works_dump, ratings_dump):
        if not max_lines and not dump_is_complete(dump, EXPECTED_DUMP_BYTES[dump.name]):
            raise FileNotFoundError(f"{dump} is missing or not fully downloaded; see the module docstring for URLs")

    editions = scan_editions_for_goodreads_ids(editions_dump, books["canonical_book_id"],
                                               max_lines=max_lines, progress_every=20_000_000, log=log)
    editions = editions.drop_duplicates("goodreads_id")           # first edition seen per Goodreads id
    subjects = read_work_subjects(works_dump, editions["ol_work_key"].dropna(), max_lines=max_lines,
                                  progress_every=20_000_000, log=log)
    ratings = read_ratings_dump(ratings_dump)

    match = (books[["work_id", "canonical_book_id"]]
             .merge(editions.rename(columns={"goodreads_id": "canonical_book_id", "title": "ol_title"}),
                    on="canonical_book_id", how="inner")
             .merge(subjects.drop(columns=["title"]).rename(columns={
                 "work_key": "ol_work_key", "subjects": "ol_subjects", "subject_count": "ol_subject_count",
                 "first_publish_date": "ol_first_publish_date"}), on="ol_work_key", how="left")
             .merge(ratings.rename(columns={"work_key": "ol_work_key", "rating_count": "ol_rating_count",
                                            "rating_mean": "ol_rating_mean"}), on="ol_work_key", how="left"))
    match["ol_subjects"] = match["ol_subjects"].apply(lambda s: s if isinstance(s, list) else [])
    match["ol_subject_count"] = match["ol_subject_count"].fillna(0).astype(int)
    match["ol_rating_count"] = match["ol_rating_count"].fillna(0).astype(int)
    match = match.drop(columns=["isbn_10"], errors="ignore")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    match.to_pickle(cache_path)
    if log:
        log(f"saved {len(match):,} matched works to {cache_path}")
    return match


def merge_open_library_match(books: pd.DataFrame, match: pd.DataFrame) -> pd.DataFrame:
    """Left-join the cached match onto the book dimension; unmatched works keep their row."""
    columns = ["work_id", "ol_work_key", "ol_subject_count", "ol_subjects", "ol_rating_count", "ol_rating_mean"]
    out = books.merge(match[columns].drop_duplicates("work_id"), on="work_id", how="left")
    out["matched_on"] = np.where(out["ol_work_key"].notna(), "goodreads_id", "none")
    out["ol_subjects"] = out["ol_subjects"].apply(lambda s: s if isinstance(s, list) else [])
    out["ol_subject_count"] = out["ol_subject_count"].fillna(0).astype(int)
    out["ol_rating_count"] = out["ol_rating_count"].fillna(0).astype(int)
    return out


def match_report(books_ol: pd.DataFrame, train_works: pd.DataFrame, floor: int = 10) -> pd.DataFrame:
    """How much of the catalogue the Goodreads-id join reached, for the works that matter most.

    Rows: every work in the dimension; works at least one training reader liked; works at or
    above the recommendable floor. Columns: how many matched, the share, and how many of the
    matches brought subjects or Open Library ratings with them.
    """
    liked_ids = set(train_works["work_id"])
    recommendable_ids = set(train_works.loc[train_works["liked"] >= floor, "work_id"])
    groups = {
        "all works": books_ol,
        "works liked by a training reader": books_ol[books_ol["work_id"].isin(liked_ids)],
        f"works with {floor}+ training likes": books_ol[books_ol["work_id"].isin(recommendable_ids)],
    }
    rows = []
    for name, frame in groups.items():
        matched = frame[frame["matched_on"] != "none"]
        rows.append({
            "works": len(frame),
            "matched to Open Library": len(matched),
            "match rate": round(len(matched) / max(len(frame), 1), 3),
            "with subjects": int((matched["ol_subject_count"] > 0).sum()),
            "with Open Library ratings": int((matched["ol_rating_count"] > 0).sum()),
            "median subjects per matched work": float(matched["ol_subject_count"].median()) if len(matched) else 0.0,
        })
    return pd.DataFrame(rows, index=list(groups))
