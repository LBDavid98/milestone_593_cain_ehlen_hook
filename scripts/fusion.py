"""Ranked fusion of the three recommenders, for Book 3.

Each recommender answers a work with its top five work_ids (the interface fixed in Book 2).
Fusion combines those three short lists by position with reciprocal rank fusion (RRF):

    fused(w) = sum over models m of  weight_m / (k + rank of w in model m)

A work a model did not return contributes nothing for that model. Positions, not scores, are
combined, because the three scores live on different scales: a share of readers (directional),
a 0-1 overlap on likes (Jaccard), and a 0-1 overlap on finished books. k = 60 is the default
from Cormack, Clarke and Buettcher (2009); a small k trusts each model's first pick, a large k
rewards works that several models return.

    recommend_fused(tables, work_id, n=5)     the project's fused recommender
    agreement_table(tables, work_ids)          how often the three top-5 lists overlap
    fusion_variants(tables)                    deeper inputs and a two-model variant, for comparison
    weight_grid(tables, holdout, ...)          a few weight settings scored on validation readers

`per_model=5` is the agreed interface: every model contributes its top five. Deeper inputs
(10, 20, 60 per model) are evaluated only to show what the agreement rule does with more
candidates. Book 3's recommender fuses the two Jaccard tables; the three-model fusion is its comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rank_fusion import fuse  # noqa: E402  (the verified reference implementation of RRF)

SEED = 593
RRF_K = 60
PER_MODEL = 5                     # the agreed interface: each model's top five
MODEL_NAMES = ["Directional (likes)", "Jaccard (likes)", "Finished together (shelves)"]
TABLE_FILES = {                   # display name -> Book 2 neighbour table
    "Directional (likes)": "neighbours_directional_liked.pkl",
    "Jaccard (likes)": "neighbours_jaccard_liked.pkl",
    "Finished together (shelves)": "neighbours_jaccard_finished.pkl",
}


# ----------------------------------------------------------------------------- the rule
def rrf_scores(rankings: dict[str, list[int]], k: int = RRF_K, weights: dict[str, float] | None = None) -> dict:
    """Weighted reciprocal rank fusion over named ranked lists; returns {work_id: fused score}.

    A plain dict implementation is used at query time because the evaluation asks tens of
    thousands of questions and rank_fusion.fuse() builds a DataFrame per call. With equal
    weights the two give the same order; check_matches_reference() asserts that.
    """
    scores: dict[int, float] = {}
    for model, ids in rankings.items():
        weight = 1.0 if weights is None else float(weights.get(model, 1.0))
        if weight == 0:
            continue
        for rank, work_id in enumerate(dict.fromkeys(int(w) for w in ids), start=1):
            scores[work_id] = scores.get(work_id, 0.0) + weight / (k + rank)
    return scores


def fuse_rankings(rankings: dict[str, list[int]], n: int, k: int = RRF_K,
                  weights: dict[str, float] | None = None) -> list[int]:
    """Top n work_ids by fused score. Ties: more models agreeing first, then the better single
    rank, then the lower work_id, the same order rank_fusion.fuse() uses."""
    scores = rrf_scores(rankings, k, weights)
    if not scores:
        return []
    appearances = {w: 0 for w in scores}
    best_rank = {w: np.inf for w in scores}
    for model, ids in rankings.items():
        if weights is not None and float(weights.get(model, 1.0)) == 0:
            continue                       # a zero-weight model takes no part in the tie-break either
        for rank, work_id in enumerate(dict.fromkeys(int(w) for w in ids), start=1):
            appearances[work_id] += 1
            best_rank[work_id] = min(best_rank[work_id], rank)
    ordered = sorted(scores, key=lambda w: (-scores[w], -appearances[w], best_rank[w], w))
    return ordered[:n]


def check_matches_reference(rankings: dict[str, list[int]], n: int = 15, k: int = RRF_K) -> bool:
    """True when fuse_rankings() returns the same order as rank_fusion.fuse() with equal weights."""
    reference = fuse(rankings, n=n, k=k)["work_id"].astype(int).tolist()
    return fuse_rankings(rankings, n=n, k=k) == reference


# ----------------------------------------------------------------------------- the recommender
def recommend_fused(tables: dict, work_id: int, n: int = 5, per_model: int = PER_MODEL, k: int = RRF_K,
                    exclude=frozenset(), weights: dict[str, float] | None = None) -> list[int]:
    """The fused recommender: ask each table for its top `per_model` works, fuse by rank, return n.

    tables   {display name: NeighbourTable} from Book 2.
    exclude  works to leave out before fusing (the reader's own seeds, same-author works, ...).
    weights  optional {display name: weight}; None means equal weights.
    """
    rankings = {name: table.recommend(work_id, per_model, exclude) for name, table in tables.items()}
    return fuse_rankings(rankings, n=n, k=k, weights=weights)


def fusion_variants(tables: dict) -> dict:
    """Named recommenders for the comparison, all with the recommend(work_id, n, exclude) signature.

    "Fused (top 5 each)" is the agreed setting. The deeper variants show what the agreement rule
    does with more candidates per model; the two-model variant drops the directional score,
    which Book 2 showed behaves like the popularity baseline.
    """
    two_jaccard = {name: tables[name] for name in ["Jaccard (likes)", "Finished together (shelves)"]}

    def make(tabs, per_model):
        return lambda w, n, exclude: recommend_fused(tabs, w, n, per_model=per_model, exclude=exclude)

    return {
        "Fused (top 5 each)": make(tables, 5),
        "Fused (top 10 each)": make(tables, 10),
        "Fused (top 20 each)": make(tables, 20),
        "Fused (top 60 each)": make(tables, 60),
        "Fused, two Jaccard models only (top 5 each)": make(two_jaccard, 5),
    }


def single_models(tables: dict) -> dict:
    """The three Book 2 recommenders under their display names, same signature as the variants."""
    return {name: (lambda t: (lambda w, n, exclude: t.recommend(w, n, exclude)))(table)
            for name, table in tables.items()}


def weight_grid(tables: dict, holdout: dict, similar_books: np.ndarray, work_ids: np.ndarray,
                popularity: pd.Series, per_model: int = PER_MODEL) -> pd.DataFrame:
    """Score a few weight settings on one holdout (validation): equal weights, the directional
    score at half weight, and the directional score at zero. Returns evaluation.evaluate_models()
    rows, one per setting."""
    import evaluation as ev
    settings = {
        "equal weights": None,
        "directional at 0.5": {"Directional (likes)": 0.5},
        "directional at 0": {"Directional (likes)": 0.0},
    }
    models = {name: (lambda ws: (lambda w, n, exclude: recommend_fused(tables, w, n, per_model=per_model,
                                                                       exclude=exclude, weights=ws)))(weights)
              for name, weights in settings.items()}
    return ev.evaluate_models(models, holdout, similar_books, work_ids, popularity)


# ----------------------------------------------------------------------------- agreement
def agreement_table(tables: dict, work_ids: np.ndarray, per_model: int = PER_MODEL, sample: int = 3000,
                    seed: int = SEED) -> pd.DataFrame:
    """How often the three models' top-`per_model` lists overlap, on a sample of seed works.

    One row per pair of models: mean Jaccard overlap of the two sets and the share of seeds with
    at least one work in common. A final row gives the share of seeds where all three models
    share at least one work. Overlap is what RRF rewards: a work returned by two models scores
    about twice one returned by a single model at the same rank.
    """
    rng = np.random.default_rng(seed)
    seeds = rng.choice(work_ids, size=min(sample, len(work_ids)), replace=False)
    names = list(tables)
    lists = {name: [set(tables[name].recommend(int(s), per_model)) for s in seeds] for name in names}
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            jaccard = [len(x & y) / len(x | y) if (x | y) else 0.0 for x, y in zip(lists[a], lists[b])]
            shared = [bool(x & y) for x, y in zip(lists[a], lists[b])]
            rows.append({"models": f"{a} vs {b}", "model a": a, "model b": b,
                         "mean overlap of top-5 sets": round(float(np.mean(jaccard)), 3),
                         "share of seeds with a common work": round(float(np.mean(shared)), 3)})
    all_three = [bool(x & y & z) for x, y, z in zip(*[lists[n] for n in names])] if len(names) == 3 else []
    rows.append({"models": "all three", "model a": "", "model b": "",
                 "mean overlap of top-5 sets": np.nan,
                 "share of seeds with a common work": round(float(np.mean(all_three)), 3) if all_three else np.nan})
    return pd.DataFrame(rows).set_index("models")


# ----------------------------------------------------------------------------- worked examples
def load_tables(processed_dir) -> dict:
    """The three Book 2 neighbour tables under their display names."""
    import cooccurrence as co
    processed_dir = Path(processed_dir)
    return {name: co.NeighbourTable.load(processed_dir / filename) for name, filename in TABLE_FILES.items()}


def main() -> None:
    """Evaluate the fused recommender and its variants on validation and test readers; print both."""
    import evaluation as ev
    import cooccurrence as co
    processed = Path(__file__).resolve().parent.parent / "data" / "processed"
    tables = load_tables(processed)
    interactions = pd.read_pickle(processed / "interactions_sample_50k.pkl")
    assignment = pd.read_pickle(processed / "user_splits.pkl")
    train_likes = pd.read_pickle(processed / "train_likes_per_work.pkl")
    similar_books = np.load(processed / "similar_books.npy")
    work_ids = co.recommendable_works(train_likes, 10)
    interactions["split"] = interactions["user_id"].map(assignment)
    popularity = ev.popularity_from(interactions[interactions["split"] == "train"])

    toy = {"a": [1, 2, 3, 4, 5], "b": [2, 3, 9, 1, 8], "c": [7, 2, 1, 6, 5]}
    assert check_matches_reference(toy), "fuse_rankings disagrees with rank_fusion.fuse on the toy lists"
    rng = np.random.default_rng(SEED)
    for _ in range(200):
        lists = {m: [int(x) for x in rng.choice(50, size=rng.integers(1, 12), replace=False)] for m in "abc"}
        assert check_matches_reference(lists), "fuse_rankings disagrees with rank_fusion.fuse"
    print("fuse_rankings matches rank_fusion.fuse on 201 random cases")

    models = {**single_models(tables), **fusion_variants(tables)}
    for split in ("validation", "test"):
        readers = assignment.index[assignment == split]
        holdout = ev.holdout_split_for_readers(interactions, readers, work_ids, seed=SEED)
        print(f"\n== {split}: {len(holdout):,} readers")
        print(ev.evaluate_models(models, holdout, similar_books, work_ids, popularity).to_string())


if __name__ == "__main__":
    main()
