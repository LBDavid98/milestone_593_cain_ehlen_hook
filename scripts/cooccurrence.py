"""The shared engine behind every Book 2 model: count co-presence once, score it three ways.

Every recommender in this project asks the same question of the training readers: for a
selected work, which other works share readers with it, and how strongly? the two
functions answer it one work at a time by scanning the whole table. This module answers it
for all works at once with one sparse matrix product, then applies a formula to the counts:

    directional_score   C[i, j] / n[i]                    the "share of the selected work's
                                                           readers who also liked the candidate"
    jaccard_score       C[i, j] / (n[i] + n[j] - C[i, j]) Jaccard: overlap of the two
                                                           reader groups
    pmi_score           log(C[i, j] * N / (n[i] * n[j]))  how much more often the pair occurs
                                                           than popularity alone predicts

where C counts the readers two works share, n counts each work's readers and N is the number
of readers. The same formula can run on three signals from the same table:

    shelved   every row (the work is on the reader's shelf)
    finished  rows with is_read == 1 (the reader marked it read)
    liked     rows rated 4 or 5

so a model is a (signal, formula) pair. Book 2 uses liked + directional and liked + Jaccard
(same numbers as the original functions, see likes_models.py) and finished + Jaccard
(the "finished together" model); shelved + Jaccard and shelved + PMI are kept for the
"what we tried" discussion.

Each build produces a NeighbourTable: the k strongest neighbours of every recommendable work,
saved as a pickle. recommend(work_id, n=5) is then a lookup, which is the function contract the
team agreed on for Book 3's rank fusion.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"

LIKE_RATINGS = (4, 5)
DEFAULT_FLOOR = 10          # training likes a work needs to be recommendable (Book 1, Section 5)
DEFAULT_K = 60              # neighbours stored per work; recommend() returns the first n that pass
DEFAULT_MIN_COUNT = 3       # pairs sharing fewer readers than this are ignored (one shared reader is noise)

# A signal is a row filter on the interactions table. Each returns a boolean mask.
SIGNALS = {
    "shelved": lambda frame: np.ones(len(frame), dtype=bool),
    "finished": lambda frame: frame["is_read"].to_numpy() == 1,
    "liked": lambda frame: frame["rating"].isin(LIKE_RATINGS).to_numpy(),
}


# ----------------------------------------------------------------------------- catalogue and matrix
def recommendable_works(train_likes_per_work: pd.Series, floor: int = DEFAULT_FLOOR) -> np.ndarray:
    """Sorted work_ids with at least `floor` likes among training readers.

    The floor comes from Book 1 (10 likes), counted on the training split only, so validation
    and test readers never influence which works may be recommended.
    """
    return np.sort(train_likes_per_work.index[train_likes_per_work >= floor].to_numpy().astype(np.int64))


def build_user_work_matrix(interactions: pd.DataFrame, work_ids: np.ndarray, signal: str):
    """Binary readers x works matrix for one signal; columns follow `work_ids` order.

    Rows outside the catalogue are dropped. Returns (csr_matrix float32, user_ids), where
    user_ids[i] is the reader behind row i. Readers with no qualifying row are absent.
    """
    keep = SIGNALS[signal](interactions) & np.isin(interactions["work_id"].to_numpy(), work_ids)
    frame = interactions[keep]
    user_ids, row = np.unique(frame["user_id"].to_numpy(), return_inverse=True)
    col = np.searchsorted(work_ids, frame["work_id"].to_numpy())
    matrix = sp.csr_matrix((np.ones(len(frame), dtype=np.float32), (row, col)), shape=(len(user_ids), len(work_ids)))
    matrix.data[:] = 1.0           # a reader-work pair counts once even if the table had duplicates
    return matrix, user_ids


def cooccurrence_counts(matrix: sp.csr_matrix, min_count: int = DEFAULT_MIN_COUNT):
    """Count the readers every pair of works shares.

    Returns (C, n, n_users): C is a works x works sparse matrix of shared-reader counts with
    counts below `min_count` zeroed (a pair resting on one or two readers is noise, as the
    Book 1 audit of the first KNN showed), n is each work's reader count, n_users the number
    of readers in the matrix. The diagonal of C is each work's own count and is never used
    as a neighbour.
    """
    counts = (matrix.T @ matrix).tocsr()
    counts.data[counts.data < min_count] = 0
    counts.eliminate_zeros()
    n = np.asarray(matrix.sum(axis=0)).ravel()
    return counts, n, matrix.shape[0]


# ----------------------------------------------------------------------------- scorers
def directional_score(counts: np.ndarray, n_rows: np.ndarray, n_cols: np.ndarray, n_users: int) -> np.ndarray:
    """The directional score: of the readers who liked the selected work (rows), the share
    who also liked the candidate (columns). Asymmetric: popular candidates score well for every
    selected work, which the Book 2 evaluation reports as popularity bias."""
    return counts / np.maximum(n_rows, 1)


def jaccard_score(counts: np.ndarray, n_rows: np.ndarray, n_cols: np.ndarray, n_users: int) -> np.ndarray:
    """Jaccard: shared readers over readers of either work. Symmetric, 0 to 1."""
    return counts / np.maximum(n_rows + n_cols - counts, 1)


def pmi_score(counts: np.ndarray, n_rows: np.ndarray, n_cols: np.ndarray, n_users: int) -> np.ndarray:
    """Pointwise mutual information: log of observed shared readers over the number expected if
    the two works were unrelated. Computed only where a pair has readers; other cells are -inf
    so they can never be chosen."""
    scores = np.full(counts.shape, -np.inf, dtype=np.float64)
    present = counts > 0
    expected = (n_rows * n_cols) / max(n_users, 1)       # broadcast to the block's shape
    expected = np.broadcast_to(expected, counts.shape)
    scores[present] = np.log(counts[present] / expected[present])
    return scores


SCORERS = {"directional": directional_score, "jaccard": jaccard_score, "pmi": pmi_score}


# ----------------------------------------------------------------------------- the neighbour table
@dataclass
class NeighbourTable:
    """The k strongest neighbours of every recommendable work, for one (signal, scorer) pair.

    neighbours[i, :] are work_ids best first, padded with -1 when a work has fewer than k
    eligible neighbours; scores[i, :] are the matching scores. recommend() is the function
    contract the team agreed on: the top n work_ids for a work, after an optional exclusion set
    (the reader's own books, same-author works, ...).
    """
    work_ids: np.ndarray
    neighbours: np.ndarray
    scores: np.ndarray
    signal: str
    scorer: str
    k: int
    min_count: int
    _index: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._index = {int(w): i for i, w in enumerate(self.work_ids)}

    def __len__(self) -> int:
        return len(self.work_ids)

    def recommend(self, work_id: int, n: int = 5, exclude=frozenset()) -> list[int]:
        """The first n neighbours of `work_id` not in `exclude`; [] when the work is unknown."""
        row = self._index.get(int(work_id))
        if row is None:
            return []
        out = []
        for candidate in self.neighbours[row]:
            if candidate < 0:
                break
            if candidate not in exclude:
                out.append(int(candidate))
                if len(out) == n:
                    break
        return out

    def neighbours_of(self, work_id: int) -> pd.DataFrame:
        """All stored neighbours of a work with their scores, best first (empty if unknown)."""
        row = self._index.get(int(work_id))
        if row is None:
            return pd.DataFrame(columns=["work_id", "score"])
        valid = self.neighbours[row] >= 0
        return pd.DataFrame({"work_id": self.neighbours[row][valid], "score": self.scores[row][valid]})

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump({"work_ids": self.work_ids, "neighbours": self.neighbours, "scores": self.scores,
                         "signal": self.signal, "scorer": self.scorer, "k": self.k, "min_count": self.min_count},
                        handle, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, path) -> "NeighbourTable":
        with open(path, "rb") as handle:
            return cls(**pickle.load(handle))


def build_neighbour_table(interactions: pd.DataFrame, work_ids: np.ndarray, signal: str, scorer: str,
                          k: int = DEFAULT_K, min_count: int = DEFAULT_MIN_COUNT, block: int = 2000,
                          log=None) -> NeighbourTable:
    """Build the neighbour table for one (signal, scorer) pair on the given interactions.

    The co-occurrence matrix is scored in blocks of `block` rows so no dense works x works
    array is ever held. Within a block: the work itself is excluded, only pairs with a kept
    co-occurrence count are eligible, and the top k by score are stored (ties broken by the
    lower work_id, which is also the original implementation's tie-break).
    """
    started = time.time()
    matrix, _ = build_user_work_matrix(interactions, work_ids, signal)
    counts, n, n_users = cooccurrence_counts(matrix, min_count)
    score = SCORERS[scorer]
    size = len(work_ids)
    neighbours = np.full((size, k), -1, dtype=np.int64)
    scores = np.zeros((size, k), dtype=np.float32)

    for start in range(0, size, block):
        stop = min(start + block, size)
        counts_block = counts[start:stop].toarray()
        block_scores = score(counts_block, n[start:stop, None], n[None, :], n_users).astype(np.float64)
        block_scores[counts_block == 0] = -np.inf                 # no shared readers: never a neighbour
        block_scores[np.arange(stop - start), np.arange(start, stop)] = -np.inf   # not itself
        # lexsort on (-score, work_id) gives best score first, lower work_id on ties
        order = np.lexsort((np.broadcast_to(work_ids, block_scores.shape), -block_scores), axis=1)[:, :k]
        chosen = np.take_along_axis(block_scores, order, axis=1)
        valid = np.isfinite(chosen)
        ids = work_ids[order]
        ids[~valid] = -1
        neighbours[start:stop] = ids
        scores[start:stop] = np.where(valid, chosen, 0).astype(np.float32)

    if log:
        log(f"{signal} + {scorer}: {size:,} works x {k} neighbours from {n_users:,} readers "
            f"in {time.time() - started:.0f}s")
    return NeighbourTable(work_ids=work_ids, neighbours=neighbours, scores=scores, signal=signal,
                          scorer=scorer, k=k, min_count=min_count)


# ----------------------------------------------------------------------------- query-time filter
def related_works(books: pd.DataFrame, work_ids: np.ndarray) -> dict[int, set[int]]:
    """For each work in the catalogue, the other catalogue works by the same author or in the
    same series, from the book dimension's `author_ids` and `series_ids` lists.

    Optional, applied by the caller as recommend(..., exclude=related[work_id] | ...). The
    Book 1 KNN audit found about a quarter of unfiltered picks were other books by the same
    author, so Book 2 reports results with and without this filter.
    """
    catalogue = set(int(w) for w in work_ids)
    rows = books[books["work_id"].isin(catalogue)][["work_id", "author_ids", "series_ids"]]
    by_key: dict[tuple, set[int]] = {}
    for work_id, authors, series in zip(rows["work_id"], rows["author_ids"], rows["series_ids"]):
        for author in authors or []:
            by_key.setdefault(("a", int(author)), set()).add(int(work_id))
        for one_series in series or []:
            by_key.setdefault(("s", int(one_series)), set()).add(int(work_id))
    related: dict[int, set[int]] = {}
    for members in by_key.values():
        if len(members) < 2:
            continue
        for work_id in members:
            related.setdefault(work_id, set()).update(members - {work_id})
    return related


# ----------------------------------------------------------------------------- build everything
BOOK2_TABLES = {   # file stem -> (signal, scorer); the first three are the models, the last two the alternatives tried
    "directional_liked": ("liked", "directional"),
    "jaccard_liked": ("liked", "jaccard"),
    "jaccard_finished": ("finished", "jaccard"),
    "jaccard_shelved": ("shelved", "jaccard"),
    "pmi_shelved": ("shelved", "pmi"),
}


def load_training_interactions() -> tuple[pd.DataFrame, np.ndarray]:
    """The training readers' rows and the recommendable catalogue, from Book 1's pickles."""
    interactions = pd.read_pickle(PROCESSED / "interactions_sample_50k.pkl")
    split = pd.read_pickle(PROCESSED / "user_splits.pkl")
    train = interactions[interactions["user_id"].map(split).eq("train").to_numpy()]
    work_ids = recommendable_works(pd.read_pickle(PROCESSED / "train_likes_per_work.pkl"))
    return train, work_ids


def main() -> None:
    train, work_ids = load_training_interactions()
    print(f"training rows {len(train):,}; recommendable works {len(work_ids):,}")
    for stem, (signal, scorer) in BOOK2_TABLES.items():
        table = build_neighbour_table(train, work_ids, signal, scorer, log=print)
        print(f"  saved {table.save(PROCESSED / f'neighbours_{stem}.pkl')}")


if __name__ == "__main__":
    main()
