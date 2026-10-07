"""Interactive "how much evidence does each book have?" explorer for Book 1.

The notebook calls two functions:

    works = summarise_works(interactions, books)              # one row per work
    chart = build_evidence_explorer(works, similar_books)     # Altair chart
    chart                                                     # renders in the notebook

Everything else in this module is a helper those two call. The chart has four
linked panels driven by one slider (the minimum number of likes a work needs):

    A  every work placed by (likes, share of raters who liked it) and counted in
       cells; hovering a cell names its most-liked books, searching a title marks it
    B  genre mix of the works above the floor; click a bar to light up a genre
    C  what the floor costs: share of works, likes and benchmark books kept
    D  Goodreads average rating against Open Library average rating for the works above the
       floor, following the genre click (needs the book table joined to the Open Library match)

Only small pre-aggregated tables are embedded in the chart, so it stays light
even though the sample has hundreds of thousands of works.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from chart_theme import (AQUA, AXIS, BLUE, BLUE_RAMP, FONT, INK, INK2, ORANGE, ORANGE_INK, SURFACE,
                         theme as _apply_theme)

# ----------------------------------------------------------------------------- settings
LIKE_RATINGS = (4, 5)          # a "like" is a 4 or 5 star rating
DEFAULT_FLOOR = 10             # where the slider starts
SEARCH_MIN_LIKES = 50          # the title search covers works with at least this many likes
TITLES_PER_CELL = 3            # most-liked titles named in a cell's tooltip
MIN_OL_RATINGS = 5             # panel D shows works with at least this many Open Library ratings
ALL_GENRES = "All genres"

GENRE_LABELS = {  # book_dimension genre columns -> labels for the chart
    "g_children": "Children", "g_comics": "Comics & graphic", "g_fantasy": "Fantasy & paranormal",
    "g_fiction": "Fiction", "g_history": "History & biography", "g_mystery": "Mystery & thriller",
    "g_nonfiction": "Non-fiction", "g_poetry": "Poetry", "g_romance": "Romance",
    "g_youngadult": "Young adult",
}
NO_GENRE = "No genre label"

# Palette: a single blue ramp for "how many works", orange only for search hits,
# and three categorical slots for the cost lines (validated colour-blind safe as a set).
LOG_TICKS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
W_MAIN, W_SIDE, H_TOP, H_BOTTOM = 560, 215, 320, 185   # fits a notebook output cell


# ----------------------------------------------------------------------------- data shaping
def summarise_works(interactions: pd.DataFrame, books: pd.DataFrame) -> pd.DataFrame:
    """Collapse user-work interactions to one row per work, with its metadata.

    interactions: columns user_id, work_id, rating (0 = shelved but not rated).
                  One row per user and work.
    books:        the book dimension, one row per work_id.

    Returns works that received at least one like, with columns
    work_id, shelved, rated, liked, like_share, title, author, year, genre,
    decade, gr_rating. Works that nobody liked cannot be recommended by a
    likes-based model, so they are left out.
    """
    rating = interactions["rating"].to_numpy()
    per_work = (
        pd.DataFrame({
            "work_id": interactions["work_id"].to_numpy(),
            "shelved": 1,
            "rated": (rating > 0).astype(np.int32),
            "liked": np.isin(rating, LIKE_RATINGS).astype(np.int32),
        })
        .groupby("work_id", sort=False)
        .sum()
        .reset_index()
    )
    per_work = per_work[per_work["liked"] >= 1].copy()
    per_work["like_share"] = (per_work["liked"] / per_work["rated"]).round(4)

    meta = pd.DataFrame({
        "work_id": books["work_id"].to_numpy(),
        "title": books["title"].fillna("").str.slice(0, 80).to_numpy(),
        "author": books["author_names"].str[0].fillna("").to_numpy(),
        "year": books["pub_year"].fillna(0).astype(int).to_numpy(),
        "genre": top_genre_label(books),
        "gr_rating": books["avg_rating"].to_numpy(),
    })
    # Open Library ratings ride along when the book table has been joined to the match (Section 6)
    for column in ("ol_rating_mean", "ol_rating_count"):
        if column in books.columns:
            meta[column] = books[column].to_numpy()
    works = per_work.merge(meta, on="work_id", how="left")
    works["genre"] = works["genre"].fillna(NO_GENRE)
    works["title"] = works["title"].fillna("")
    works["author"] = works["author"].fillna("")
    works["decade"] = decade_label(works["year"].fillna(0).astype(int).to_numpy())
    return works


def top_genre_label(books: pd.DataFrame) -> np.ndarray:
    """Pick each work's strongest genre column; works with no label get NO_GENRE."""
    cols = [c for c in books.columns if c.startswith("g_")]
    weights = books[cols].to_numpy()
    labels = np.array([GENRE_LABELS.get(c, c) for c in cols])[weights.argmax(axis=1)]
    labels[weights.max(axis=1) == 0] = NO_GENRE
    return labels


def decade_label(years: np.ndarray) -> np.ndarray:
    """Bucket publication years into decades; 0 and implausible years become 'unknown'."""
    in_range = (years >= 1900) & (years <= 2019)
    return np.where(in_range, (years // 10 * 10).astype(str).astype(object) + "s",
                    np.where((years > 0) & (years < 1900), "pre-1900", "unknown"))


def floor_values(max_likes: int) -> np.ndarray:
    """Slider stops: every value 1-30, then log-spaced integers up to the most-liked work.

    Log spacing lets one slider cover the whole log x-axis while staying fine-grained
    where the team's real choices lie (floors of 2 to 20).
    """
    coarse = np.round(np.geomspace(30, max_likes, 80)).astype(int)
    return np.unique(np.r_[np.arange(1, 31), coarse])


def bin_works_for_density(works: pd.DataFrame) -> pd.DataFrame:
    """Count works in (likes x like_share) cells, once for all works and once per genre.

    Likes are binned on integer, roughly log-spaced edges so each cell is a whole
    number of likes. Like share is binned in 5% steps. Each cell also carries the
    titles of its most-liked works so a hover always names real books.
    Returns genre, x0, x1, y0, y1, works, top_titles.
    """
    max_likes = int(works["liked"].max()) + 1
    edges = np.unique(np.round(np.geomspace(1, max_likes, 50)).astype(int))
    edges = np.unique(np.r_[np.arange(1, 11), edges[edges > 10], max_likes])
    y_edges = np.linspace(0, 1, 21)

    placed = works[["genre", "title", "liked", "like_share"]].copy()
    placed["xi"] = np.searchsorted(edges, placed["liked"].to_numpy(), side="right") - 1
    placed["yi"] = np.minimum(np.searchsorted(y_edges, placed["like_share"].to_numpy(), side="right") - 1, 19)
    placed = placed.sort_values("liked", ascending=False)

    def cells_for(frame, label):
        grouped = frame.groupby(["xi", "yi"], sort=False)
        cells = grouped.size().rename("works").reset_index()
        titles = grouped["title"].agg(
            lambda t: "; ".join(x[:40] for x in t.head(TITLES_PER_CELL) if x)).rename("top_titles")
        cells = cells.merge(titles.reset_index(), on=["xi", "yi"])
        cells["genre"] = label
        return cells

    cells = pd.concat([cells_for(placed, ALL_GENRES)]
                      + [cells_for(group, genre) for genre, group in placed.groupby("genre")], ignore_index=True)
    cells["x0"], cells["x1"] = edges[cells["xi"]], edges[cells["xi"] + 1]
    cells["y0"], cells["y1"] = y_edges[cells["yi"]].round(2), y_edges[cells["yi"] + 1].round(2)
    return cells[["genre", "x0", "x1", "y0", "y1", "works", "top_titles"]]


def count_works_by_genre_decade_floor(works: pd.DataFrame, floors: np.ndarray) -> pd.DataFrame:
    """For every slider position, how many works survive in each genre and decade.

    The chart filters this table to the current floor, so panels B and D update
    without the browser having to hold every work.
    """
    rows = []
    for (genre, decade), group in works.groupby(["genre", "decade"]):
        likes = np.sort(group["liked"].to_numpy())
        surviving = len(likes) - np.searchsorted(likes, floors, side="left")
        rows += [(genre, decade, int(f), int(n)) for f, n in zip(floors, surviving) if n > 0]
    return pd.DataFrame(rows, columns=["genre", "decade", "floor", "works"])


def cost_of_minimum_likes(works: pd.DataFrame, similar_books: np.ndarray,
                          floors: np.ndarray) -> pd.DataFrame:
    """Share of works, likes and benchmark books that remain at each floor.

    A benchmark book is a work with a Goodreads "readers also enjoyed" list. It
    stays testable at a floor if it and at least one book on its list both have
    that many likes. Returns long-form rows: floor, measure, share.
    """
    likes = np.sort(works["liked"].to_numpy())
    cumulative = np.r_[0, np.cumsum(likes)]
    first_kept = np.searchsorted(likes, floors, side="left")
    works_kept = (len(likes) - first_kept) / len(likes)
    likes_kept = (cumulative[-1] - cumulative[first_kept]) / cumulative[-1]

    likes_by_work = pd.Series(works["liked"].to_numpy(), index=works["work_id"].to_numpy())
    seed_likes = likes_by_work.reindex(similar_books[:, 0]).fillna(0).to_numpy()
    target_likes = likes_by_work.reindex(similar_books[:, 1]).fillna(0).to_numpy()
    # the best link a seed has is the one whose weaker side has the most likes
    best_link = pd.Series(np.minimum(seed_likes, target_likes)).groupby(similar_books[:, 0]).max()
    best_link = np.sort(best_link[best_link >= 1].to_numpy())
    bench_kept = (len(best_link) - np.searchsorted(best_link, floors, side="left")) / len(best_link)

    wide = pd.DataFrame({"floor": floors, "Likes kept": likes_kept, "Works kept": works_kept,
                         "Benchmark books testable": bench_kept})
    return wide.melt("floor", var_name="measure", value_name="share")


# ----------------------------------------------------------------------------- the chart
def build_evidence_explorer(works: pd.DataFrame, similar_books: np.ndarray,
                            sample_readers: int | None = None) -> alt.VConcatChart:
    """Assemble the four linked panels. `works` comes from summarise_works()."""
    alt.data_transformers.disable_max_rows()
    floors = floor_values(int(works["liked"].max()))
    density = bin_works_for_density(works)
    by_floor = count_works_by_genre_decade_floor(works, floors)
    cost = cost_of_minimum_likes(works, similar_books, floors)
    searchable = works.loc[works["liked"] >= SEARCH_MIN_LIKES,
                           ["title", "author", "genre", "year", "liked", "rated", "like_share", "gr_rating"]].copy()
    searchable["like_share"] = searchable["like_share"].round(3)
    searchable["gr_rating"] = searchable["gr_rating"].round(2)
    x_max = float(works["liked"].max()) * 1.15

    # shared controls: the floor slider, a title search box, and a genre click.
    # HTML sliders are linear, so the slider picks an index into the log-spaced
    # floor values and a second parameter turns that index into the floor itself.
    floor_index = alt.param(name="floor_i", value=int(np.searchsorted(floors, DEFAULT_FLOOR)),
                            bind=alt.binding_range(min=0, max=len(floors) - 1, step=1,
                                                   name="Minimum likes a work needs (slider position; "
                                                        "the value is printed on the chart)  "))
    floor = alt.param(name="floor", expr=f"[{','.join(map(str, floors))}][floor_i]")
    search = alt.param(name="search", value="",
                       bind=alt.binding(input="search", placeholder="type part of a title", name="Find a title  "))
    genre = alt.selection_point(name="genre", fields=["genre"], on="click", clear="dblclick")
    at_floor = "datum.floor == floor"
    matches_search = "length(search) > 1 && indexof(lower(datum.title), lower(search)) >= 0"
    # the clicked genre (if any) picks which set of density cells is shown
    chosen_genre = ("isValid(genre) && isValid(genre.genre) ? datum.genre == genre.genre[0] "
                    f": datum.genre == '{ALL_GENRES}'")

    panel_a = _evidence_panel(density, searchable, floor, matches_search, chosen_genre, x_max)
    panel_b = _genre_panel(by_floor, genre, at_floor)
    panel_c = _cost_panel(cost, floor, at_floor, x_max)
    panel_d = _rating_panel(works, genre)

    single_like_share = (works["liked"] == 1).mean()
    subtitle = (f"{len(works):,} works with at least one 4-5 star rating"
                + (f" from {sample_readers:,} sampled readers" if sample_readers else "")
                + f". {single_like_share:.0%} of them have exactly one.")

    chart = (
        alt.vconcat(alt.hconcat(panel_a, panel_b, spacing=36),
                    alt.hconcat(panel_c, panel_d, spacing=36), spacing=30)
        .add_params(floor_index, floor, search)
        .resolve_scale(color="independent")
        .properties(title=alt.TitleParams("How much evidence does each book have?", subtitle=subtitle,
                                          fontSize=20, subtitleFontSize=13, anchor="start", offset=18))
    )
    return _apply_theme(chart)


def _evidence_panel(density, searchable, floor, matches_search, chosen_genre, x_max):
    """Panel A: counted cells across the whole range (darker = more works), the floor rule,
    and orange markers on the works whose title matches the search box."""
    x_scale = alt.Scale(type="log", domain=[1, x_max], nice=False)
    x_axis = alt.Axis(values=LOG_TICKS, format="~s", title="Readers in the sample who liked the work (log scale)")
    y_axis = alt.Axis(format="%", title="Share of its raters who gave 4-5 stars")

    hovered = alt.selection_point(name="hovered", on="mouseover", clear="mouseout", nearest=False,
                                  fields=["x0", "y0"], empty=False)
    cells = (
        alt.Chart(density).transform_filter(chosen_genre)
        .mark_rect(stroke=SURFACE, strokeWidth=0.6)
        .encode(x=alt.X("x0:Q", scale=x_scale, axis=x_axis), x2="x1:Q",
                y=alt.Y("y0:Q", scale=alt.Scale(domain=[0, 1]), axis=y_axis), y2="y1:Q",
                color=alt.Color("works:Q", scale=alt.Scale(type="log", range=BLUE_RAMP),
                                legend=alt.Legend(title="Works in the cell", orient="bottom", direction="horizontal",
                                                  gradientLength=180, format="~s", tickCount=4)),
                # cells entirely left of the floor are what the rule removes
                opacity=alt.condition("datum.x1 <= floor", alt.value(0.25), alt.value(1)),
                tooltip=[alt.Tooltip("works:Q", title="Works here", format=","),
                         alt.Tooltip("top_titles:N", title="Most liked"),
                         alt.Tooltip("x0:Q", title="Likes from"), alt.Tooltip("x1:Q", title="to below"),
                         alt.Tooltip("y0:Q", title="Share liked from", format=".0%")])
        .add_params(hovered)
    )
    # The highlight is its own layer, drawn after every cell, so all four edges of the outline
    # show. Drawn as the cell's own stroke, the neighbours painted later covered two edges and
    # the outline looked shifted away from the pointer.
    highlight = (
        alt.Chart(density).transform_filter(chosen_genre).transform_filter(hovered)
        .mark_rect(fill=None, stroke=INK, strokeWidth=2)
        .encode(x=alt.X("x0:Q", scale=x_scale), x2="x1:Q", y=alt.Y("y0:Q"), y2="y1:Q")
    )
    hits = alt.Chart(searchable).transform_filter(matches_search).encode(
        x=alt.X("liked:Q", scale=x_scale), y="like_share:Q", tooltip=_book_tooltips())
    hit_marks = hits.mark_point(size=110, color=ORANGE, strokeWidth=2.5, opacity=1, clip=True)
    hit_summary = (  # one summary line in the corner, so clustered hits never print over each other
        hits.transform_joinaggregate(matches="count()")
        .transform_window(rank="rank()", sort=[alt.SortField("liked", order="descending")])
        .transform_filter("datum.rank == 1")
        .transform_calculate(summary="datum.matches + (datum.matches == 1 ? ' match' : ' matches') + "
                                     f"' with {SEARCH_MIN_LIKES}+ likes. Most liked: ' + datum.title")
        .mark_text(align="left", baseline="bottom", fontSize=11.5, fontWeight=600, color=ORANGE_INK, font=FONT,
                   limit=600)
        .encode(x=alt.value(8), y=alt.value(H_TOP - 8), text="summary:N")
    )
    floor_rule, floor_label = _floor_rule(floor, x_scale)
    return alt.layer(cells, highlight, hit_marks, hit_summary, floor_rule, floor_label).properties(
        width=W_MAIN, height=H_TOP,
        title=alt.TitleParams("Most liked works have only one or two likes", subtitle=[
            "Each cell counts the works at that many likes and that share of approval; "
            "hover to see its most-liked books.",
            "Left of the floor line is what a minimum-likes rule removes. Click a genre on the right to refocus."]))


def _genre_panel(by_floor, genre, at_floor):
    """Panel B: works above the floor per genre; clicking a bar selects that genre."""
    base = (alt.Chart(by_floor).transform_filter(at_floor)
            .transform_aggregate(works="sum(works)", groupby=["genre"]))
    by_count = alt.EncodingSortField(field="works", order="descending")  # same sort on both layers
    bars = (
        base.mark_bar(color=BLUE, cornerRadiusEnd=4, height=17)
        .encode(y=alt.Y("genre:N", sort=by_count, title=None, axis=alt.Axis(labelLimit=150)),
                x=alt.X("works:Q", title="Works at or above the floor", axis=alt.Axis(format="~s", tickCount=4)),
                opacity=alt.condition(genre, alt.value(1), alt.value(0.28)),
                tooltip=[alt.Tooltip("genre:N", title="Genre"), alt.Tooltip("works:Q", title="Works", format=",")])
        .add_params(genre)
    )
    labels = base.mark_text(align="left", dx=4, fontSize=11, color=INK2, font=FONT).encode(
        y=alt.Y("genre:N", sort=by_count), x="works:Q", text=alt.Text("works:Q", format=","))
    return alt.layer(bars, labels).properties(
        width=W_SIDE, height=H_TOP,
        title=alt.TitleParams("Genre mix of what survives",
                              subtitle=["Click a bar to light up that genre.", "Double-click to clear."]))


def _cost_panel(cost, floor, at_floor, x_max):
    """Panel C: three cost curves against the floor, with the current floor marked."""
    measures = ["Likes kept", "Works kept", "Benchmark books testable"]
    x_scale = alt.Scale(type="log", domain=[1, x_max], nice=False)
    base = alt.Chart(cost).encode(
        x=alt.X("floor:Q", scale=x_scale, title="Minimum likes a work needs (log scale)",
                axis=alt.Axis(values=LOG_TICKS, format="~s")),
        y=alt.Y("share:Q", scale=alt.Scale(domain=[0, 1]), title="Share remaining",
                axis=alt.Axis(format="%", tickCount=5)),
        color=alt.Color("measure:N", scale=alt.Scale(domain=measures, range=[BLUE, ORANGE, AQUA]),
                        legend=alt.Legend(title=None, orient="top", direction="horizontal", labelLimit=300,
                                          symbolType="stroke", symbolStrokeWidth=3)))
    lines = base.mark_line(strokeWidth=2)
    now = base.transform_filter(at_floor)
    markers = now.mark_circle(size=80, opacity=1, stroke=SURFACE, strokeWidth=2)
    # value labels: "Works kept" sits below its marker, the other two above, so the
    # two curves that converge at higher floors never print on top of each other
    label = dict(fontSize=12, fontWeight=600, font=FONT)
    values_above = (now.transform_filter("datum.measure != 'Works kept'")
                    .mark_text(align="left", dx=9, dy=-9, **label)
                    .encode(text=alt.Text("share:Q", format=".0%"), color=alt.value(INK)))
    values_below = (now.transform_filter("datum.measure == 'Works kept'")
                    .mark_text(align="left", dx=9, dy=13, **label)
                    .encode(text=alt.Text("share:Q", format=".0%"), color=alt.value(INK)))
    floor_rule, _ = _floor_rule(floor, x_scale)
    return alt.layer(lines, floor_rule, markers, values_above, values_below).properties(
        width=W_MAIN, height=H_BOTTOM,
        title=alt.TitleParams("What the floor costs", subtitle=["A low floor drops most works yet keeps most likes."]))


def _rating_panel(works, genre):
    """Panel D: Goodreads average rating against Open Library average rating, one dot per work.

    Only works with at least MIN_OL_RATINGS Open Library ratings are drawn, because a mean of one or
    two stars-out-of-five is noise. Dots follow the slider (works at or above the floor) and the
    genre click. The dashed diagonal marks exact agreement between the two communities.
    """
    works = works.copy()
    for column in ("ol_rating_mean", "ol_rating_count"):       # absent when books lack the Open Library join
        if column not in works.columns:
            works[column] = np.nan
    pts = works[(works["ol_rating_count"].fillna(0) >= MIN_OL_RATINGS) & (works["gr_rating"] > 0)]
    pts = pts[["title", "author", "genre", "liked", "gr_rating", "ol_rating_mean", "ol_rating_count"]].copy()
    pts["ol_rating_mean"] = pts["ol_rating_mean"].round(2)
    corr = pts["gr_rating"].corr(pts["ol_rating_mean"]) if len(pts) > 2 else float("nan")
    axis = alt.Scale(domain=[1, 5])
    dots = (
        alt.Chart(pts).transform_filter("datum.liked >= floor").transform_filter(genre)
        .mark_circle(size=16, color=BLUE, opacity=0.35, stroke=SURFACE, strokeWidth=0.4)
        .encode(x=alt.X("gr_rating:Q", scale=axis, title="Goodreads average (to 2017)"),
                y=alt.Y("ol_rating_mean:Q", scale=axis, title="Open Library average (2026)"),
                tooltip=[alt.Tooltip("title:N", title="Title"), alt.Tooltip("author:N", title="Author"),
                         alt.Tooltip("gr_rating:Q", title="Goodreads", format=".2f"),
                         alt.Tooltip("ol_rating_mean:Q", title="Open Library", format=".2f"),
                         alt.Tooltip("ol_rating_count:Q", title="Open Library ratings", format=","),
                         alt.Tooltip("liked:Q", title="Readers who liked it", format=",")])
    )
    diagonal = alt.Chart(pd.DataFrame({"v": [1, 5]})).mark_line(color=INK2, strokeDash=[4, 4], strokeWidth=1).encode(
        x=alt.X("v:Q", scale=axis), y=alt.Y("v:Q", scale=axis))
    return alt.layer(dots, diagonal).properties(
        width=W_SIDE, height=H_BOTTOM,
        title=alt.TitleParams("Do the two communities agree?", subtitle=[
            f"{len(pts):,} works with {MIN_OL_RATINGS}+ Open Library ratings; correlation {corr:.2f}.",
            "Follows the floor and the chosen genre. Dashed line = same rating."]))


def _floor_rule(floor, x_scale):
    """A vertical rule at the slider value, plus a small label above it."""
    anchor = alt.Chart(pd.DataFrame({"k": [0]})).transform_calculate(f=floor.name)
    rule = anchor.mark_rule(color=INK, strokeWidth=1.5).encode(x=alt.X("f:Q", scale=x_scale))
    label = (anchor.transform_calculate(t="'floor = ' + floor + (floor == 1 ? ' like' : ' likes')")
             .mark_text(align="left", dx=6, dy=-4, baseline="bottom", fontSize=11, fontWeight=600, color=INK, font=FONT)
             .encode(x=alt.X("f:Q", scale=x_scale), y=alt.value(14), text="t:N"))
    return rule, label


def _book_tooltips():
    return [alt.Tooltip("title:N", title="Title"), alt.Tooltip("author:N", title="Author"),
            alt.Tooltip("genre:N", title="Genre"), alt.Tooltip("year:Q", title="Published", format="d"),
            alt.Tooltip("liked:Q", title="Readers who liked it", format=","),
            alt.Tooltip("rated:Q", title="Readers who rated it", format=","),
            alt.Tooltip("like_share:Q", title="Share giving 4-5 stars", format=".0%"),
            alt.Tooltip("gr_rating:Q", title="Goodreads average", format=".2f")]


def save_explorer_html(chart, path: str) -> None:
    """Write a standalone HTML page (controls on top) that opens in any browser."""
    chart.save(path, embed_options={"actions": False})
    css = (
        "<style>body{background:%s;margin:0;font-family:%s;color:%s}"
        # align-items:flex-start matters: with the default (stretch) the canvas is scaled to the
        # window width and Vega's hit-testing, which ignores CSS scaling, picks the wrong cell
        ".vega-embed{display:flex;flex-direction:column;align-items:flex-start}"
        ".vega-embed canvas,.vega-embed svg{flex:none}"
        ".vega-bindings{order:-1;display:flex;gap:32px;flex-wrap:wrap;padding:20px 28px 0;font-size:14px}"
        ".vega-bind{display:flex;align-items:center;gap:8px}.vega-bind-name{font-weight:600}"
        ".vega-bind input[type=range]{width:260px}.vega-bind input[type=range]+span{display:none}"
        ".vega-bind input[type=search]{width:220px;padding:5px 8px;border:1px solid %s;border-radius:6px;font:inherit}"
        "</style>"
    ) % (SURFACE, FONT, INK, AXIS)
    with open(path) as f:
        html = f.read()
    with open(path, "w") as f:
        f.write(html.replace("</head>", css + "</head>"))
