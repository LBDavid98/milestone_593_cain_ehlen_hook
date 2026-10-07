"""One evaluation protocol for every recommender in Books 2 and 3.

A recommender is any function recommend(work_id, n, exclude) -> list of work_ids, best first.
That is the contract the team agreed on, so the likes models, the finished-together model, the baselines and
Book 3's fused recommender are all scored by the same code here.

Two targets, both built only from data the models never saw:

1. Held-out readers (validation or test split). For each reader, a random fifth of the works
   they liked is hidden; the rest are the seeds. The recommender is asked about every seed, the
   lists are combined with reciprocal rank fusion (the same formula Book 3 uses), and the top n
   are checked against the hidden works: precision, recall, hit rate and NDCG.
2. Goodreads' own "readers also enjoyed" lists (similar_books.npy): for a sample of seed works,
   does the recommender's top n contain any work Goodreads lists for it?

Plus two descriptive measures: coverage (how many distinct works a recommender ever shows) and
the popularity of what it recommends compared with the seeds, because a model can score well by
recommending bestsellers to everyone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rank_fusion import fuse  # noqa: E402  (the verified RRF; same k = 60 everywhere)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"

LIKE_RATINGS = (4, 5)
SEED = 593
RRF_K = 60

# chart chrome shared with catalogue_charts.py / evidence_explorer.py
INK, INK2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE = "#2a78d6"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


# ----------------------------------------------------------------------------- held-out readers
def holdout_split_for_readers(interactions: pd.DataFrame, reader_ids, work_ids: np.ndarray,
                              like_ratings=LIKE_RATINGS, hide_share: float = 0.2, min_likes: int = 5,
                              seed: int = SEED) -> dict[int, tuple[list[int], set[int]]]:
    """For each reader, split the works they liked (within the catalogue) into seeds and hidden.

    Readers with fewer than `min_likes` liked catalogue works are skipped: one hidden work out
    of two or three is too little to score. The hidden share is at least one work. The shuffle
    is seeded per reader so the split is the same on every run and does not depend on order.
    Returns {user_id: (seeds, hidden)}.
    """
    catalogue = set(int(w) for w in work_ids)
    liked = interactions[interactions["rating"].isin(like_ratings) & interactions["user_id"].isin(reader_ids)]
    liked = liked[liked["work_id"].isin(catalogue)]
    holdout = {}
    rng = np.random.default_rng(seed)
    for user_id, works in liked.groupby("user_id")["work_id"]:
        items = sorted(int(w) for w in works.unique())
        if len(items) < min_likes:
            continue
        rng_user = np.random.default_rng([seed, int(user_id)])      # reproducible per reader
        rng_user.shuffle(items)
        n_hidden = max(1, int(round(len(items) * hide_share)))
        holdout[int(user_id)] = (items[n_hidden:], set(items[:n_hidden]))
    del rng
    return holdout


def rrf_merge(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    """Reciprocal rank fusion of several ranked lists: score(w) = sum over lists of 1 / (k + rank).

    The same formula as rank_fusion.fuse (rank starts at 1, k = 60, ties broken by more lists
    agreeing, then the better single rank, then work_id). Implemented with plain dicts because
    the evaluation calls it tens of thousands of times; `check_rrf_matches_fuse` confirms the two
    agree.
    """
    score: dict[int, float] = {}
    count: dict[int, int] = {}
    best: dict[int, int] = {}
    for ranked in rankings:
        for rank, work_id in enumerate(dict.fromkeys(ranked), start=1):
            score[work_id] = score.get(work_id, 0.0) + 1.0 / (k + rank)
            count[work_id] = count.get(work_id, 0) + 1
            best[work_id] = min(best.get(work_id, rank), rank)
    return sorted(score, key=lambda w: (-score[w], -count[w], best[w], w))


def check_rrf_matches_fuse(rankings: list[list[int]], n: int = 20) -> bool:
    """True when rrf_merge and rank_fusion.fuse return the same top-n order for these lists."""
    named = {f"list{i}": r for i, r in enumerate(rankings) if r}
    depth = max((len(r) for r in named.values()), default=0)
    reference = [int(w) for w in fuse(named, n=depth, k=RRF_K)["work_id"]][:n] if named else []
    return rrf_merge(rankings)[:n] == reference


def recommend_for_reader(recommend_fn, seeds: list[int], n: int, exclude: set[int]) -> list[int]:
    """Combine the recommender's lists for a reader's seeds with reciprocal rank fusion.

    Each seed contributes its full list (the recommender is asked for 60, the depth the
    neighbour tables store); the seeds themselves and anything in `exclude` are removed before
    the top n is taken. Book 3's fused recommender applies the identical formula.
    """
    rankings = [recommend_fn(seed_work, 60, frozenset()) for seed_work in seeds]
    blocked = set(seeds) | set(exclude)
    return [w for w in rrf_merge(rankings) if w not in blocked][:n]


def _ndcg(hits: list[bool], n_relevant: int) -> float:
    gains = sum(1.0 / np.log2(rank + 2) for rank, hit in enumerate(hits) if hit)
    ideal = sum(1.0 / np.log2(rank + 2) for rank in range(min(n_relevant, len(hits))))
    return gains / ideal if ideal else 0.0


def evaluate_readers(recommend_fn, holdout: dict, n: int = 10, exclude_fn=None) -> dict:
    """Score a recommender on held-out readers.

    exclude_fn(seed_works) -> set, optional, lets the caller drop related works (same author,
    same series) before scoring. Returns precision@5, precision@10, recall@10, hit@10, ndcg@10,
    averaged over readers, and the number of readers scored.
    """
    p5 = p10 = r10 = hit = ndcg = 0.0
    for seeds, hidden in holdout.values():
        exclude = exclude_fn(seeds) if exclude_fn else set()
        recs = recommend_for_reader(recommend_fn, seeds, n, exclude)
        hits = [w in hidden for w in recs] + [False] * (n - len(recs))
        p5 += sum(hits[:5]) / 5
        p10 += sum(hits[:10]) / 10
        r10 += sum(hits[:10]) / len(hidden)
        hit += float(any(hits))
        ndcg += _ndcg(hits[:10], len(hidden))
    count = max(len(holdout), 1)
    return {"precision@5": p5 / count, "precision@10": p10 / count, "recall@10": r10 / count,
            "hit@10": hit / count, "ndcg@10": ndcg / count, "readers": len(holdout)}


# ----------------------------------------------------------------------------- Goodreads' lists
def goodreads_lists(similar_books: np.ndarray, work_ids: np.ndarray) -> dict[int, set[int]]:
    """Goodreads' "readers also enjoyed" links restricted to the catalogue: {work: set of works}."""
    catalogue = set(int(w) for w in work_ids)
    frame = pd.DataFrame(similar_books, columns=["a", "b"])
    frame = frame[frame["a"].isin(catalogue) & frame["b"].isin(catalogue)]
    return {int(a): set(int(x) for x in b) for a, b in frame.groupby("a")["b"]}


def goodreads_hit_rate(recommend_fn, similar_books: np.ndarray, work_ids: np.ndarray, n: int = 10,
                       max_seeds: int = 3000, seed: int = SEED, lists: dict | None = None) -> float:
    """Share of sampled seed works whose top-n contains at least one of Goodreads' listed works."""
    lists = lists if lists is not None else goodreads_lists(similar_books, work_ids)
    seeds = np.array(sorted(lists))
    rng = np.random.default_rng(seed)
    if len(seeds) > max_seeds:
        seeds = rng.choice(seeds, size=max_seeds, replace=False)
    hits = sum(any(w in lists[int(s)] for w in recommend_fn(int(s), n, frozenset())) for s in seeds)
    return hits / max(len(seeds), 1)


# ----------------------------------------------------------------------------- descriptive measures
def catalogue_stats(recommend_fn, holdout: dict, popularity: pd.Series, n: int = 10) -> dict:
    """Coverage (distinct works ever recommended) and median popularity of recommendations and seeds.

    popularity: work_id -> number of training readers who shelved it. A recommender whose
    recommendations are far more popular than the seeds is pushing bestsellers.
    """
    shown: set[int] = set()
    rec_pop: list[float] = []
    seed_pop: list[float] = []
    for seeds, _ in holdout.values():
        recs = recommend_for_reader(recommend_fn, seeds, n, set())
        shown.update(recs)
        rec_pop += [float(popularity.get(w, 0)) for w in recs]
        seed_pop += [float(popularity.get(w, 0)) for w in seeds]
    return {"coverage": len(shown),
            "median popularity (recs)": float(np.median(rec_pop)) if rec_pop else 0.0,
            "median popularity (seeds)": float(np.median(seed_pop)) if seed_pop else 0.0}


def recommendation_counts(recommend_fn, holdout: dict, n: int = 10) -> pd.Series:
    """How often each work is recommended across all held-out readers (top n per reader).

    Used for the concentration view: a model that returns the same few works to everyone has a
    steep curve; a model that draws on the whole catalogue has a flat one.
    """
    counts = {}
    for seeds, _hidden in holdout.values():
        for work in recommend_for_reader(recommend_fn, seeds, n, set(seeds)):
            counts[work] = counts.get(work, 0) + 1
    return pd.Series(counts, dtype="int64").sort_values(ascending=False)


def popularity_from(train_interactions: pd.DataFrame) -> pd.Series:
    """Training readers per work (any shelving), the popularity measure used throughout."""
    return train_interactions.groupby("work_id")["user_id"].nunique()


# ----------------------------------------------------------------------------- baselines
def baselines(train_interactions: pd.DataFrame, work_ids: np.ndarray, books: pd.DataFrame, seed: int = SEED) -> dict:
    """Three recommenders every model has to beat.

    most_popular        the most-shelved catalogue works, the same list for every seed
    random              a random catalogue sample, different per seed but reproducible
    same_genre_popular  the most-shelved works sharing the seed's top genre
    Each follows the recommend(work_id, n, exclude) contract.
    """
    catalogue = np.array(sorted(int(w) for w in work_ids))
    popularity = popularity_from(train_interactions).reindex(catalogue).fillna(0)
    by_popularity = catalogue[np.argsort(-popularity.to_numpy(), kind="stable")]

    genre_columns = [c for c in books.columns if c.startswith("g_")]
    meta = books[books["work_id"].isin(set(catalogue.tolist()))].set_index("work_id")
    weights = meta[genre_columns].to_numpy()
    top_genre = pd.Series(np.where(weights.max(axis=1) > 0, weights.argmax(axis=1), -1), index=meta.index)
    popular_by_genre = {g: [w for w in by_popularity if top_genre.get(w, -1) == g] for g in range(len(genre_columns))}

    def take(pool, work_id, n, exclude):
        # walk the ranked pool and stop as soon as n items pass; pools are 43K long, so no full scans
        out = []
        for w in pool:
            if w != work_id and w not in exclude:
                out.append(int(w))
                if len(out) == n:
                    break
        return out

    def most_popular(work_id, n, exclude):
        return take(by_popularity, work_id, n, exclude)

    def random_recommender(work_id, n, exclude):
        rng = np.random.default_rng([seed, int(work_id)])
        picks = rng.choice(catalogue, size=min(n + len(exclude) + 1, len(catalogue)), replace=False)
        return take(picks, work_id, n, exclude)

    def same_genre_popular(work_id, n, exclude):
        genre = top_genre.get(int(work_id), -1)
        pool = popular_by_genre.get(genre, []) if genre >= 0 else by_popularity
        return take(pool, work_id, n, exclude)

    return {"most popular": most_popular, "random": random_recommender, "same-genre popular": same_genre_popular}


# ----------------------------------------------------------------------------- one table for all models
def evaluate_models(models: dict, holdout: dict, similar_books: np.ndarray, work_ids: np.ndarray,
                    popularity: pd.Series, exclude_fn=None, log=None) -> pd.DataFrame:
    """One row per recommender: held-out metrics, Goodreads hit rate, coverage and popularity."""
    lists = goodreads_lists(similar_books, work_ids)
    rows = []
    for name, recommend_fn in models.items():
        row = {"model": name}
        row.update(evaluate_readers(recommend_fn, holdout, exclude_fn=exclude_fn))
        row["Goodreads hit@10"] = goodreads_hit_rate(recommend_fn, similar_books, work_ids, lists=lists)
        row.update(catalogue_stats(recommend_fn, holdout, popularity))
        rows.append(row)
        if log:
            log(f"scored {name}")
    table = pd.DataFrame(rows).set_index("model")
    return table.round({"precision@5": 4, "precision@10": 4, "recall@10": 4, "hit@10": 4, "ndcg@10": 4,
                        "Goodreads hit@10": 4})


# ----------------------------------------------------------------------------- the chart
METRICS = ["precision@5", "precision@10", "hit@10", "Goodreads hit@10"]


def performance_chart(results: pd.DataFrame, title: str = "How the recommenders compare") -> alt.VConcatChart:
    """Grouped bars, one panel per metric with a bar per model, plus a popularity panel.

    One hue throughout: the bars are the same kind of thing (a model's score), so colour would
    only repeat what the x-axis labels say. The popularity panel marks the seeds' median with a
    rule; bars above it recommend books more widely read than what readers started from.
    Book 3 calls this with a fused row added to `results`.
    """
    order = list(results.index)
    long = results.reset_index().melt(id_vars="model", value_vars=METRICS, var_name="metric", value_name="value")
    bars = (
        alt.Chart(long).mark_bar(color=BLUE, cornerRadiusEnd=3)
        .encode(x=alt.X("model:N", sort=order, title=None, axis=alt.Axis(labelAngle=-30, labelLimit=140)),
                y=alt.Y("value:Q", title="score", axis=alt.Axis(format=".0%")),
                tooltip=[alt.Tooltip("model:N"), alt.Tooltip("metric:N"), alt.Tooltip("value:Q", format=".3f")])
        .properties(width=150, height=180)
        .facet(column=alt.Column("metric:N", sort=METRICS, title=None, header=alt.Header(labelFontWeight="bold")))
    )
    pop = results.reset_index()[["model", "median popularity (recs)", "median popularity (seeds)"]]
    seed_median = float(pop["median popularity (seeds)"].iloc[0]) if len(pop) else 0.0
    popularity = (
        alt.Chart(pop).mark_bar(color=BLUE, cornerRadiusEnd=3)
        .encode(x=alt.X("model:N", sort=order, title=None, axis=alt.Axis(labelAngle=-30, labelLimit=140)),
                y=alt.Y("median popularity (recs):Q", title="median training readers of recommended works",
                        scale=alt.Scale(type="symlog")),
                tooltip=[alt.Tooltip("model:N"), alt.Tooltip("median popularity (recs):Q", format=",.0f")])
    )
    rule = alt.Chart(pd.DataFrame({"y": [seed_median]})).mark_rule(color=INK2, strokeDash=[4, 4]).encode(y="y:Q")
    label = alt.Chart(pd.DataFrame({"y": [seed_median], "t": [f"seeds' median: {seed_median:,.0f}"]})).mark_text(
        align="left", dx=4, dy=-6, fontSize=11, color=INK2, font=FONT).encode(y="y:Q", text="t:N", x=alt.value(0))
    popularity_panel = alt.layer(popularity, rule, label).properties(
        width=max(150, 60 * len(order)), height=180,
        title=alt.TitleParams("Popularity of what each model recommends",
                              subtitle=["Dashed line: median popularity of the seed works."]))
    return (alt.vconcat(bars, popularity_panel, spacing=28)
            .properties(title=alt.TitleParams(title, fontSize=16, anchor="start"))
            .configure(background=SURFACE, font=FONT)
            .configure_view(stroke=None)
            .configure_axis(gridColor=GRID, domainColor=AXIS, tickColor=AXIS, labelColor=INK2, titleColor=INK2,
                            labelFontSize=11, titleFontSize=12, titleFontWeight="normal")
            .configure_title(color=INK, subtitleColor=INK2, fontSize=14, subtitleFontSize=11.5, anchor="start")
            .configure_header(labelColor=INK, labelFontSize=12))


# ----------------------------------------------------------------------------- run on validation
def main() -> None:
    import time
    from cooccurrence import NeighbourTable, BOOK2_TABLES, load_training_interactions

    started = time.time()
    interactions = pd.read_pickle(PROCESSED / "interactions_sample_50k.pkl")
    split = pd.read_pickle(PROCESSED / "user_splits.pkl")
    books = pd.read_pickle(PROCESSED / "book_dimension.pkl")
    similar_books = np.load(PROCESSED / "similar_books.npy")
    train, work_ids = load_training_interactions()
    popularity = popularity_from(train)

    validation_readers = split.index[split == "validation"].to_numpy()
    rng = np.random.default_rng(SEED)
    validation_readers = rng.choice(validation_readers, size=min(2000, len(validation_readers)), replace=False)
    holdout = holdout_split_for_readers(interactions, validation_readers, work_ids)
    print(f"validation readers with 5+ liked catalogue works: {len(holdout):,} ({time.time() - started:.0f}s)",
          flush=True)

    models = {}
    for stem in BOOK2_TABLES:
        table = NeighbourTable.load(PROCESSED / f"neighbours_{stem}.pkl")
        models[stem] = table.recommend
    first = NeighbourTable.load(PROCESSED / "neighbours_jaccard_liked.pkl")
    for seeds, _ in list(holdout.values())[:25]:
        assert check_rrf_matches_fuse([first.recommend(w, 60) for w in seeds]), "rrf_merge disagrees with fuse"
    print("rrf_merge matches rank_fusion.fuse on 25 readers", flush=True)
    models.update(baselines(train, work_ids, books))
    results = evaluate_models(models, holdout, similar_books, work_ids, popularity,
                              log=lambda m: print(m, flush=True))
    pd.set_option("display.width", 200)
    print(results.to_string())
    print(f"done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
