"""The two likes-based recommendation methods, kept as written, with the wrappers Book 2 needs.

Section 1 is the original code from `final-staging/recommendation_methods.ipynb`, copied byte for
byte (cells 4, 6 and 8 of that notebook): a helper that counts the readers two works share,
the directional score and the Jaccard score. The original toy example (cell 10) and the results it
checked by hand (cells 12 and 13) are also carried over, as data, so the fast version can be
tested against them. `verify_verbatim()` re-reads that notebook and confirms nothing changed.

Section 2 adds the fast path. The original functions scan the whole interactions table for every question, which
is fine for one click on the dashboard but too slow to evaluate (tens of thousands of
questions). The team agreed every model would expose `recommend(work_id, n=5)` returning five
work ids, and that fusion in Book 3 would combine those lists. So:

    recommend_directional(table, work_id, n=5)    fast: reads a precomputed NeighbourTable
    recommend_jaccard(table, work_id, n=5)        fast: same
    recommend_directional_reference(df, work_id)  slow: calls the original function unchanged
    recommend_jaccard_reference(df, work_id)      slow: same
    check_against_toy_example(...)                the fast tables reproduce the hand-checked orders
    check_equivalence(...)                        the fast tables match the original functions on real seeds
    method_discussion()                           the original explanations, verbatim, for the notebook

The precomputed tables come from `scripts/cooccurrence.py`, which applies the two formulas
(shared / readers of the selected work; shared / readers of either work) to every pair at once.
The original method is unchanged; only the order of computation differs, and the two checks prove it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_NOTEBOOK = Path(__file__).resolve().parent.parent / "final-staging" / "recommendation_methods.ipynb"
DISCUSSION_FILE = Path(__file__).resolve().parent / "likes_models_discussion.json"   # original Markdown
VERBATIM_CELLS = (4, 6, 8)       # code cells of the original notebook reproduced below, in this order

# =============================================================================================
# Section 1 - the original code, copied verbatim from final-staging/recommendation_methods.ipynb
# (cells 4, 6, 8). Do not edit. verify_verbatim() checks these against the notebook.
# =============================================================================================


def count_shared_readers(interactions_df, selected_item_id, item_column="work_id"):
    """Count the readers each other item shares with the selected item."""

    selected_readers = interactions_df.loc[
        interactions_df[item_column] == selected_item_id,
        "user_id",
    ].unique()

    if len(selected_readers) == 0:
        raise ValueError("The selected item has no readers in the input data.")

    candidate_interactions = interactions_df.loc[
        interactions_df["user_id"].isin(selected_readers)
        & (interactions_df[item_column] != selected_item_id),
        [item_column, "user_id"],
    ]

    shared_reader_counts = (
        candidate_interactions.groupby(item_column)["user_id"]
        .nunique()
        .rename("shared_reader_count")
        .reset_index()
    )

    return shared_reader_counts, len(selected_readers)


def directional_recommendations(
    interactions_df,
    selected_item_id,
    item_column="work_id",
    top_n=5,
):
    """Return ranked directional recommendations from positive interactions."""

    recommendations, selected_reader_count = count_shared_readers(
        interactions_df, selected_item_id, item_column
    )

    recommendations["directional_score"] = (
        recommendations["shared_reader_count"] / selected_reader_count * 100
    )

    # Use the item ID to give tied scores a consistent order.
    recommendations = recommendations.sort_values(
        ["directional_score", item_column],
        ascending=[False, True],
    )

    return recommendations.head(top_n).reset_index(drop=True)


def jaccard_recommendations(
    interactions_df,
    selected_item_id,
    item_column="work_id",
    top_n=5,
):
    """Return ranked Jaccard recommendations from positive interactions."""

    recommendations, selected_reader_count = count_shared_readers(
        interactions_df, selected_item_id, item_column
    )

    # Count all readers of each candidate, including readers outside
    # the selected work's reader group.
    total_reader_counts = (
        interactions_df.groupby(item_column)["user_id"].nunique()
    )

    recommendations["candidate_reader_count"] = (
        recommendations[item_column].map(total_reader_counts)
    )

    recommendations["jaccard_score"] = (
        recommendations["shared_reader_count"]
        / (
            selected_reader_count
            + recommendations["candidate_reader_count"]
            - recommendations["shared_reader_count"]
        )
    )

    recommendations = recommendations.sort_values(
        ["jaccard_score", item_column],
        ascending=[False, True],
    )

    return recommendations.head(top_n).reset_index(drop=True)


# The toy example (cell 10 of the original notebook) and the results it worked out by hand
# (cells 11-13): four works, five readers, work A selected. Kept as data for the checks below.
EXPECTED_TOY_RESULTS = {
    # method: (work order, shared reader counts, scores)
    "directional": (["B", "C", "D"], [2, 2, 1], [50.0, 50.0, 25.0]),
    "jaccard": (["C", "B", "D"], [2, 2, 1], [0.5, 0.4, 0.2]),
}


def toy_example() -> pd.DataFrame:
    """The original five-reader, four-work example as a DataFrame (user_id, work_id)."""
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "work_id": [
                "A", "B", "C",
                "A", "B",
                "A", "C",
                "A", "D",
                "B", "D",
            ],
        }
    )


def verify_verbatim(notebook_path=SOURCE_NOTEBOOK) -> bool:
    """Confirm the three functions above are byte for byte the code cells of the original notebook.

    Reads cells 4, 6 and 8 of `recommendation_methods.ipynb` and compares them with the source
    of this file between the Section 1 and Section 2 markers. Raises AssertionError on any
    difference, so a stray edit to the original code is caught by the notebook's first cell that calls it.
    """
    with open(notebook_path) as handle:
        cells = ["".join(c["source"]) for c in json.load(handle)["cells"]]
    own = Path(__file__).read_text()
    for index in VERBATIM_CELLS:
        assert cells[index] in own, f"cell {index} of the source notebook differs from this module"
    return True


# =============================================================================================
# Section 2 - ours: wrappers to the team contract, the reference path, and the checks
# =============================================================================================


def _work_ids(frame: pd.DataFrame, n: int, exclude) -> list:
    """The first n work ids of a result frame, skipping excluded ones (seeds, same-series, ...)."""
    ids = [w for w in frame["work_id"].tolist() if w not in exclude]
    return ids[:n]


def recommend_directional(table, work_id, n: int = 5, exclude=frozenset()) -> list:
    """Top n work ids by the directional score, from a precomputed NeighbourTable.

    `table` is built by cooccurrence.build_neighbour_table(..., signal="liked", scorer="directional").
    The group contract is n=5; `exclude` lets the caller drop the seed books or filtered works
    while still returning n results, which is why the table keeps 60 neighbours.
    """
    return list(table.recommend(work_id, n=n, exclude=exclude))


def recommend_jaccard(table, work_id, n: int = 5, exclude=frozenset()) -> list:
    """Top n work ids by the Jaccard score, from a precomputed NeighbourTable."""
    return list(table.recommend(work_id, n=n, exclude=exclude))


def recommend_directional_reference(interactions_df: pd.DataFrame, work_id, n: int = 5) -> list:
    """the original directional_recommendations(), unchanged, returning only the work ids.

    Slow path: scans the whole table per call (about half a second on the training split).
    Suitable for a single dashboard click and for checking the fast path.
    `interactions_df` must hold likes only (rating 4-5), as the original notebook's input did.
    """
    return _work_ids(directional_recommendations(interactions_df, work_id, top_n=n), n, frozenset())


def recommend_jaccard_reference(interactions_df: pd.DataFrame, work_id, n: int = 5) -> list:
    """the original jaccard_recommendations(), unchanged, returning only the work ids."""
    return _work_ids(jaccard_recommendations(interactions_df, work_id, top_n=n), n, frozenset())


def likes_only(interactions: pd.DataFrame, like_ratings=(4, 5)) -> pd.DataFrame:
    """The rows the original functions expect: one row per reader and liked work (rating 4 or 5)."""
    liked = interactions[interactions["rating"].isin(like_ratings)]
    return liked[["user_id", "work_id"]].reset_index(drop=True)


TOY_LETTER_TO_ID = {"A": 1, "B": 2, "C": 3, "D": 4}   # the engine keys works by integer id


def toy_example_tables(k: int = 3):
    """Build the two fast tables from the original toy example.

    The engine keys works by integer id, so A-D become 1-4 for the build and are mapped back
    for the comparison. The original example has no ratings (every row is a like), so a rating of 5 is
    attached to satisfy the "liked" signal. min_count=1 because the original pairs share one reader; k=3
    because work A has exactly three other works (the engine requires k <= works - 1).
    """
    from cooccurrence import build_neighbour_table   # lazy: this module must import without the engine
    toy = toy_example()
    frame = pd.DataFrame({"user_id": toy["user_id"], "work_id": toy["work_id"].map(TOY_LETTER_TO_ID),
                          "is_read": 1, "rating": 5})
    work_ids = np.array(sorted(TOY_LETTER_TO_ID.values()))
    return (build_neighbour_table(frame, work_ids, signal="liked", scorer="directional", k=k, min_count=1),
            build_neighbour_table(frame, work_ids, signal="liked", scorer="jaccard", k=k, min_count=1))


def check_against_toy_example(table_directional=None, table_jaccard=None) -> pd.DataFrame:
    """The fast tables must reproduce the orders and scores worked out by hand in the source notebook.

    With no arguments the tables are built from the original toy example (toy_example_tables()). The original tie
    rule, ties broken by work id ascending, puts B before C in the directional result; the
    engine applies the same rule. The original directional score is a percentage and the engine's is a
    fraction, so the engine's value is multiplied by 100 before comparing. Raises
    AssertionError on any mismatch; returns the comparison as a table for the notebook.
    """
    if table_directional is None or table_jaccard is None:
        table_directional, table_jaccard = toy_example_tables()
    id_to_letter = {v: k for k, v in TOY_LETTER_TO_ID.items()}
    rows = []
    for name, table in (("directional", table_directional), ("jaccard", table_jaccard)):
        expected_order, _, expected_scores = EXPECTED_TOY_RESULTS[name]
        result = table.neighbours_of(TOY_LETTER_TO_ID["A"])
        ids = [id_to_letter.get(int(w), w) for w in result["work_id"]]
        scores = [float(x) for x in result["score"]]
        if name == "directional":
            scores = [x * 100 for x in scores]
        assert ids[:3] == expected_order, f"{name}: engine order {ids[:3]} != the original {expected_order}"
        assert np.allclose(scores[:3], expected_scores, atol=1e-6), f"{name}: scores {scores[:3]} != {expected_scores}"
        rows.append({"method": name, "the original order": expected_order, "engine order": ids[:3],
                     "the original scores": expected_scores, "engine scores": [round(x, 4) for x in scores[:3]]})
    return pd.DataFrame(rows).set_index("method")


def check_equivalence(table, reference_fn, interactions_df: pd.DataFrame, work_ids, n_seeds: int = 200,
                      seed: int = 593, top_n: int = 10) -> pd.DataFrame:
    """Compare the fast table with the original function on sampled seed works.

    `interactions_df` is the likes-only frame the original functions take (see likes_only()). The table
    passed here must have been built with min_count=1, because the original functions keep every pair,
    even those sharing a single reader; a production table with min_count=3 would differ on
    exactly those low-evidence neighbours and the comparison would measure the filter, not the
    arithmetic. Returns one row per seed with the two lists, whether they are identical, and
    the overlap, plus a summary row. Ties in the original results are ordered by work id ascending; the
    engine uses the same rule, so lists should be identical.
    """
    rng = np.random.default_rng(seed)
    sampled = rng.choice(np.asarray(list(work_ids)), size=min(n_seeds, len(work_ids)), replace=False)
    rows = []
    for work_id in sampled:
        fast = list(table.recommend(work_id, n=top_n))
        slow = reference_fn(interactions_df, work_id, n=top_n)
        rows.append({"work_id": work_id, "fast": fast, "reference": slow, "identical": fast == slow,
                     "overlap": len(set(fast) & set(slow)) / max(len(slow), 1)})
    table_out = pd.DataFrame(rows)
    summary = pd.DataFrame([{"work_id": "all seeds", "fast": "", "reference": "",
                             "identical": table_out["identical"].mean(), "overlap": table_out["overlap"].mean()}])
    return pd.concat([table_out, summary], ignore_index=True).set_index("work_id")


def summarise_equivalence(check: pd.DataFrame) -> pd.DataFrame:
    """One-row summary of check_equivalence(): seeds compared, how many matched exactly, mean
    overlap, and the first seed that differed (if any) so a mismatch can be inspected. Keeps the
    notebook output short; the per-seed frame is still there to look at."""
    per_seed = check.drop(index="all seeds", errors="ignore")
    identical = per_seed["identical"].astype(bool)
    differing = per_seed.index[~identical].tolist()
    return pd.DataFrame([{
        "seed works compared": len(per_seed),
        "identical top-10 lists": int(identical.sum()),
        "share identical": round(float(identical.mean()), 3),
        "mean overlap": round(float(per_seed["overlap"].astype(float).mean()), 3),
        "first seed that differs": differing[0] if differing else "none",
    }]).set_index("seed works compared")


def inspect_source(method: str) -> str:
    """Source of the reference implementation, for display in the notebook: "directional" or "jaccard"."""
    import inspect
    target = {"directional": directional_recommendations, "jaccard": jaccard_recommendations}[method]
    return inspect.getsource(target).rstrip()


def method_discussion() -> dict:
    """the original Markdown explanations from recommendation_methods.ipynb, verbatim, keyed by step.

    The notebook displays these with attribution next to the code they describe, so her
    reasoning stays in the original explanations.
    """
    with open(DISCUSSION_FILE) as handle:
        return json.load(handle)
