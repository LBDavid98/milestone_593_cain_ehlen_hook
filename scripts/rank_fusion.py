#!/usr/bin/env python3
"""Combine several recommenders' ranked lists with reciprocal rank fusion.

Each model submits an ordered list of work_ids, best first. The models score
on different scales (Jaccard 0-1, PMI unbounded, ratings 1-5), so scores are
never compared. Only positions are:

    fused(w) = sum over models of 1 / (k + rank of w in that model)

rank starts at 1. A work a model did not return contributes nothing for that
model. k = 60 is the default from Cormack, Clarke & Buettcher (2009). A small
k trusts each model's top picks; a large k flattens the curve and rewards
works that several models agree on.

N is how deep every list is read and how many fused results come back. When
it is not given it defaults to the shortest submitted list, so a model that
returned only 3 recommendations makes the fused answer a top 3. Empty lists
count as "no submission" (the model had nothing for this query) and do not
drag N to zero.
"""
from __future__ import annotations

import pandas as pd

DEFAULT_K = 60


def default_n(rankings: dict[str, list[int]]) -> int:
    """The shortest non-empty submitted list, after duplicates are dropped."""
    lengths = [len(dict.fromkeys(ids)) for ids in rankings.values() if len(ids)]
    return min(lengths) if lengths else 0


def fuse(rankings: dict[str, list[int]], n: int | None = None,
         k: int = DEFAULT_K) -> pd.DataFrame:
    """Fuse ranked lists into one top-N table.

    rankings  {model name: [work_id, ...]} best first. A work_id repeated
              within one list keeps its first (best) position.
    n         depth read from each list and number of rows returned.
              None = the shortest non-empty submitted list.
    k         RRF damping constant.

    Returns one row per work, best first: fused rank, work_id, fused score,
    how many models returned it, and its rank in each model (NaN if absent).
    Ties on score break on more models agreeing, then the better single
    rank, then work_id, so the output is deterministic.
    """
    if n is None:
        n = default_n(rankings)
    if n < 0:
        raise ValueError("n must be >= 0")
    if k < 0:
        raise ValueError("k must be >= 0")

    ranks: dict[int, dict[str, int]] = {}
    for model, ids in rankings.items():
        for rank, work_id in enumerate(list(dict.fromkeys(ids))[:n], start=1):
            ranks.setdefault(int(work_id), {})[model] = rank

    columns = ["rank", "work_id", "score", "models"] + list(rankings)
    if not ranks or n == 0:
        return pd.DataFrame(columns=columns)

    rows = []
    for work_id, by_model in ranks.items():
        rows.append({"work_id": work_id,
                     "score": sum(1.0 / (k + r) for r in by_model.values()),
                     "models": len(by_model),
                     "best": min(by_model.values()),
                     **{m: by_model.get(m) for m in rankings}})

    table = (pd.DataFrame(rows)
             .sort_values(["score", "models", "best", "work_id"],
                          ascending=[False, False, True, True])
             .head(n)
             .drop(columns="best")
             .reset_index(drop=True))
    table.insert(0, "rank", range(1, len(table) + 1))
    for m in rankings:
        table[m] = table[m].astype("Int64")
    return table[columns]


if __name__ == "__main__":
    demo = {"jaccard": [11, 12, 13, 14, 15],
            "pmi":     [13, 21, 11, 22, 23],
            "ratings": [21, 13, 31]}
    print(f"default N = {default_n(demo)}")
    print(fuse(demo).to_string(index=False))
