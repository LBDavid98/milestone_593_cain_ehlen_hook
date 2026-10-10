"""Bake Book 3's recommender output into app/catalogue.json, the only data the app ships with.

For every recommendable work: title, first author, training likes, the ten Goodreads genre weights,
and the top five from the fused recommender (the two Jaccard models, as Book 3's `recommend()`) and
from each single model, all with same-author and same-series works removed.

    python3.10 app/build_data.py      (from the repository root, after Books 1-3 have run)
"""

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

import book_dimension as bd  # noqa: E402
import cooccurrence as co    # noqa: E402
import fusion as fu          # noqa: E402

PROCESSED = PROJECT / "data" / "processed"
OUT = Path(__file__).resolve().parent / "catalogue.json"
MIN_LIKES, TOP_N, RRF_K = 10, 5, 60
SINGLES = {"jaccard": "Jaccard (likes)", "finished": "Finished together (shelves)",
           "directional": "Directional (likes)"}


def main():
    books = pd.read_pickle(PROCESSED / "book_dimension.pkl")
    train_likes = pd.read_pickle(PROCESSED / "train_likes_per_work.pkl")
    work_ids = co.recommendable_works(train_likes, floor=MIN_LIKES)
    tables = fu.load_tables(PROCESSED)
    jaccard_tables = {name: tables[name] for name in ["Jaccard (likes)", "Finished together (shelves)"]}
    related = co.related_works(books, work_ids)

    position = {int(w): i for i, w in enumerate(work_ids)}
    lists = {"fused": [], **{key: [] for key in SINGLES}}
    for work_id in map(int, work_ids):
        exclude = frozenset(related.get(work_id, set()))
        lists["fused"].append([position[w] for w in fu.recommend_fused(
            jaccard_tables, work_id, n=TOP_N, per_model=TOP_N, k=RRF_K, exclude=exclude)])
        for key, name in SINGLES.items():
            lists[key].append([position[w] for w in tables[name].recommend(work_id, TOP_N, exclude)])

    meta = books.set_index("work_id").loc[work_ids]
    payload = {
        "genres": bd.GENRE_LABELS,
        "title": meta["title"].tolist(),
        "author": [names[0] if len(names) else "" for names in meta["author_names"]],
        "likes": [int(train_likes.get(w, 0)) for w in work_ids],
        "genre_weights": meta[bd.GENRE_COLUMNS].astype(int).values.tolist(),
        **lists,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    print(f"{len(work_ids):,} works -> {OUT.name} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
