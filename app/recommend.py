"""The only thing the screen calls. Loads Book 3's baked output once at startup."""

import json
from pathlib import Path

DATA = json.loads((Path(__file__).resolve().parent / "catalogue.json").read_text())
TITLES, AUTHORS, LIKES = DATA["title"], DATA["author"], DATA["likes"]
GENRES = DATA["genres"]
SEARCH_TEXT = [f"{t} {a}".lower() for t, a in zip(TITLES, AUTHORS)]
RRF_K, TOP_N = 60, 5

MODES = {
    "fused": "Panel: ranked fusion (both Jaccard models)",
    "jaccard": "Single model: Jaccard on likes",
    "finished": "Single model: finished together",
    "directional": "Single model: directional on likes",
}


def search(term: str, skip=(), limit: int = 8) -> list[int]:
    """Catalogue positions whose title or author contains the term, most-liked first."""
    term = term.strip().lower()
    if len(term) < 2:
        return []
    hits = [i for i, text in enumerate(SEARCH_TEXT) if term in text and i not in skip]
    return sorted(hits, key=lambda i: -LIKES[i])[:limit]


def recommend(shelf: list[int], mode: str = "fused", n: int = TOP_N) -> list[int]:
    """Each shelf book's top five under `mode`, fused across the shelf by reciprocal rank."""
    score, count = {}, {}
    for book in shelf:
        for rank, work in enumerate(DATA[mode][book], start=1):
            if work in shelf:
                continue
            score[work] = score.get(work, 0) + 1 / (RRF_K + rank)
            count[work] = count.get(work, 0) + 1
    return sorted(score, key=lambda w: (-score[w], -count[w], -LIKES[w]))[:n]


def reasons(pick: int, shelf: list[int], mode: str) -> list[tuple[int, int]]:
    """(shelf book, rank of the pick in that book's list), best rank first."""
    found = [(book, DATA[mode][book].index(pick) + 1) for book in shelf if pick in DATA[mode][book]]
    return sorted(found, key=lambda pair: pair[1])


def genre_shares(book: int) -> list[int]:
    """Share of the book's Goodreads genre shelvings in each of the ten genres, in percent."""
    weights = DATA["genre_weights"][book]
    total = sum(weights) or 1
    return [round(100 * w / total) for w in weights]
