"""The "finished together" model for Book 2, co-presence among books readers finished.

Definition. Two works are related by how many readers finished both, measured as Jaccard
similarity: readers who marked both as read, divided by readers who marked either as read.
Only the `is_read` flag is used; no rating enters the score. Shelving a book records an
intention, finishing it records an action, and the trials showed the action is the better
signal for predicting what held-out readers liked.

The precomputed table comes from `scripts/cooccurrence.py`:

    build_neighbour_table(train, work_ids, signal="finished", scorer="jaccard", k=60, min_count=3)

and this module adds the wrapper the team agreed on, `recommend(work_id, n=5)` returning five
work ids, plus `describe()` for the notebook's model card.
"""

from __future__ import annotations

# What was tried before settling on this model (2,000 validation readers, training split,
# seed 593, 43,203 recommendable works, 60 neighbours stored, pairs sharing fewer than 3 readers
# dropped). Kept here so the notebook's "what we tried" note and the model card agree.
TRIALS = {
    "A. PMI on shelf co-presence": {
        "P@5": 0.033, "hit@10": 0.22, "Goodreads hit@10": 0.42,
        "median popularity of recommendations": 35, "coverage (works shown)": 5331},
    "B. Three-step walk (RP3, alpha 0.6)": {
        "P@5": 0.157, "hit@10": 0.56, "Goodreads hit@10": 0.07,
        "median popularity of recommendations": 10073, "coverage (works shown)": 372},
    "C. Finished together (Jaccard on is_read)": {
        "P@5": 0.237, "hit@10": 0.74, "Goodreads hit@10": 0.38,
        "median popularity of recommendations": 3778, "coverage (works shown)": 3952},
    "reference: Jaccard on all shelvings": {
        "P@5": 0.212, "hit@10": 0.74, "Goodreads hit@10": 0.45,
        "median popularity of recommendations": 3608, "coverage (works shown)": 3965},
}
CHOSEN = "C. Finished together (Jaccard on is_read)"


def recommend_finished_together(table, work_id, n: int = 5, exclude=frozenset()) -> list:
    """Top n work ids by finished-together Jaccard, from a precomputed NeighbourTable.

    `exclude` lets the caller drop seed books or same-series works and still get n results,
    which is why the table stores 60 neighbours per work.
    """
    return list(table.recommend(work_id, n=n, exclude=exclude))


def describe() -> str:
    """One paragraph for the model card: what the score is, what it does not use, and how it was chosen."""
    chosen = TRIALS[CHOSEN]
    reference = TRIALS["reference: Jaccard on all shelvings"]
    pmi = TRIALS["A. PMI on shelf co-presence"]
    walk = TRIALS["B. Three-step walk (RP3, alpha 0.6)"]
    return (
        "Finished together: Jaccard similarity on co-presence among the works a reader marked as read. "
        "Readers who finished both works, divided by readers who finished either. No rating is used. "
        f"Chosen from three candidates on validation readers: it reached P@5 {chosen['P@5']:.3f} against "
        f"{reference['P@5']:.3f} for the same formula on all shelvings, {pmi['P@5']:.3f} for PMI, and "
        f"{walk['P@5']:.3f} for a three-step walk that recommended only {walk['coverage (works shown)']} "
        "distinct works to 2,000 readers."
    )
