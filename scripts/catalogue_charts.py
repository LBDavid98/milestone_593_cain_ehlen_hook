"""Small static charts for Book 1's narrative sections.

Each function takes the data frame the notebook already has and returns an Altair chart,
so the notebook cell is one line and the chart's design decisions are documented here.

    concentration_curve(books)   Visualization 2: how few works hold most of the ratings
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

# Same palette and chrome as evidence_explorer.py so the notebook's charts read as one set.
INK, INK2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE = "#2a78d6"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def concentration_curve(books: pd.DataFrame, points: int = 400) -> alt.LayerChart:
    """Cumulative share of all Goodreads ratings held by the most-rated works.

    Works are ranked by `ratings_count`. The curve answers "what share of the catalogue do
    readers actually rate?", which is the fact that makes a minimum-evidence floor (Section 5)
    necessary. The 1.5 million works are thinned to `points` evenly spaced ranks for drawing;
    the share at each rank is exact.
    """
    counts = np.sort(books["ratings_count"].to_numpy())[::-1].astype(np.int64)
    cumulative = np.cumsum(counts) / counts.sum()
    n = len(counts)
    # log-spaced ranks so the steep start of the curve is drawn in detail
    ranks = np.unique(np.round(np.geomspace(1, n, points)).astype(int))
    curve = pd.DataFrame({
        "share_of_works": ranks / n,
        "share_of_ratings": cumulative[ranks - 1],
    })

    def share_at(pct):  # share of ratings held by the top pct of works
        return float(cumulative[max(int(n * pct) - 1, 0)])

    callouts = pd.DataFrame({
        "share_of_works": [0.01, 0.10],
        "share_of_ratings": [share_at(0.01), share_at(0.10)],
        "label": [f"top 1% of works hold {share_at(0.01):.0%} of ratings",
                  f"top 10% hold {share_at(0.10):.0%}"],
    })

    x = alt.X("share_of_works:Q", scale=alt.Scale(type="log", domain=[1 / n, 1], nice=False),
              axis=alt.Axis(format="%", values=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
                            title="Share of works, most-rated first (log scale)"))
    y = alt.Y("share_of_ratings:Q", scale=alt.Scale(domain=[0, 1]),
              axis=alt.Axis(format="%", title="Cumulative share of all ratings"))
    line = alt.Chart(curve).mark_line(color=BLUE, strokeWidth=2).encode(
        x=x, y=y,
        tooltip=[alt.Tooltip("share_of_works:Q", title="Share of works", format=".2%"),
                 alt.Tooltip("share_of_ratings:Q", title="Share of ratings", format=".1%")])
    marks = alt.Chart(callouts).mark_circle(size=70, color=BLUE, stroke=SURFACE, strokeWidth=2).encode(x=x, y=y)
    labels = alt.Chart(callouts).mark_text(align="left", baseline="top", dx=8, dy=6, fontSize=11.5, color=INK,
                                           font=FONT).encode(
        x=x, y=y, text="label:N")

    return (alt.layer(line, marks, labels)
            .properties(width=620, height=300,
                        title=alt.TitleParams("A small share of works holds most of the ratings",
                                              subtitle=f"{n:,} works ranked by Goodreads ratings count"))
            .configure(background=SURFACE, font=FONT)
            .configure_view(stroke=None)
            .configure_axis(gridColor=GRID, domainColor=AXIS, tickColor=AXIS, labelColor=INK2, titleColor=INK2,
                            labelFontSize=11, titleFontSize=12, titleFontWeight="normal")
            .configure_title(color=INK, subtitleColor=INK2, fontSize=14, subtitleFontSize=11.5, anchor="start"))


def _theme(chart):
    """Shared quiet chrome for the static charts."""
    return (chart.configure(background=SURFACE, font=FONT)
            .configure_view(stroke=None)
            .configure_axis(gridColor=GRID, domainColor=AXIS, tickColor=AXIS, labelColor=INK2, titleColor=INK2,
                            labelFontSize=11, titleFontSize=12, titleFontWeight="normal")
            .configure_title(color=INK, subtitleColor=INK2, fontSize=14, subtitleFontSize=11.5, anchor="start")
            .configure_legend(labelColor=INK2, titleColor=INK2))


LIKE_BANDS = [(1, 1, "1"), (2, 4, "2-4"), (5, 9, "5-9"), (10, 24, "10-24"), (25, 99, "25-99"), (100, None, "100+")]


def open_library_coverage_chart(books_ol: pd.DataFrame, train_works: pd.DataFrame) -> alt.Chart:
    """Visualization 3: share of works matched to Open Library, by how many training readers liked them.

    Works are grouped into bands of training likes. The question the chart answers is whether
    the join reaches the works a recommender will actually show (the right-hand bands) even if
    it misses obscure ones.
    """
    likes = train_works.set_index("work_id")["liked"]
    frame = books_ol[["work_id", "matched_on"]].copy()
    frame["liked"] = frame["work_id"].map(likes).fillna(0).astype(int)
    frame = frame[frame["liked"] >= 1]
    labels = [label for _, _, label in LIKE_BANDS]
    edges = [low for low, _, _ in LIKE_BANDS] + [np.inf]
    frame["band"] = pd.cut(frame["liked"], bins=edges, labels=labels, right=False)
    summary = (frame.groupby("band", observed=True)
               .agg(works=("work_id", "size"), matched=("matched_on", lambda m: int((m != "none").sum())))
               .reset_index())
    summary["match_rate"] = summary["matched"] / summary["works"]

    bars = alt.Chart(summary).mark_bar(color=BLUE, cornerRadiusEnd=4, width=38).encode(
        x=alt.X("band:N", sort=labels, title="Training readers who liked the work", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("match_rate:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"),
                title="Share of works matched to Open Library"),
        tooltip=[alt.Tooltip("band:N", title="Likes"), alt.Tooltip("works:Q", title="Works", format=","),
                 alt.Tooltip("matched:Q", title="Matched", format=","),
                 alt.Tooltip("match_rate:Q", title="Match rate", format=".1%")])
    text = alt.Chart(summary).mark_text(dy=-8, fontSize=11, color=INK2, font=FONT).encode(
        x=alt.X("band:N", sort=labels), y="match_rate:Q", text=alt.Text("match_rate:Q", format=".0%"))
    return _theme(alt.layer(bars, text).properties(
        width=420, height=260,
        title=alt.TitleParams("Open Library match rate by how widely a work is liked",
                              subtitle=f"{len(frame):,} works liked by at least one training reader")))


def rating_agreement_chart(books_ol: pd.DataFrame, min_ol_ratings: int = 5) -> alt.LayerChart:
    """Visualization 4: Goodreads average rating against Open Library average rating, per matched work.

    Only works with at least `min_ol_ratings` Open Library ratings are shown, because a mean of
    one or two stars-out-of-five is noise. Works are counted in 0.1-star cells; darker cells hold
    more works. The diagonal marks exact agreement.
    """
    frame = books_ol[(books_ol["ol_rating_count"] >= min_ol_ratings) & (books_ol["avg_rating"] > 0)]
    frame = frame[["avg_rating", "ol_rating_mean"]].dropna()
    step = 0.1
    cells = frame.assign(gx=(frame["avg_rating"] // step) * step, oy=(frame["ol_rating_mean"] // step) * step)
    cells = cells.groupby(["gx", "oy"]).size().rename("works").reset_index()
    cells["gx2"], cells["oy2"] = cells["gx"] + step, cells["oy"] + step
    corr = frame["avg_rating"].corr(frame["ol_rating_mean"])
    mean_gap = (frame["ol_rating_mean"] - frame["avg_rating"]).mean()

    heat = alt.Chart(cells).mark_rect(stroke=SURFACE, strokeWidth=0.5).encode(
        x=alt.X("gx:Q", scale=alt.Scale(domain=[1, 5]), title="Goodreads average rating (UCSD, to 2017)"),
        x2="gx2:Q",
        y=alt.Y("oy:Q", scale=alt.Scale(domain=[1, 5]), title="Open Library average rating (2026 dump)"),
        y2="oy2:Q",
        color=alt.Color("works:Q", scale=alt.Scale(type="log", range=["#cde2fb", "#0d366b"]),
                        legend=alt.Legend(title="Works", orient="right", format="~s")),
        tooltip=[alt.Tooltip("works:Q", title="Works", format=","),
                 alt.Tooltip("gx:Q", title="Goodreads", format=".1f"),
                 alt.Tooltip("oy:Q", title="Open Library", format=".1f")])
    diagonal = alt.Chart(pd.DataFrame({"v": [1, 5]})).mark_line(color=INK2, strokeDash=[4, 4], strokeWidth=1).encode(
        x="v:Q", y="v:Q")
    return _theme(alt.layer(heat, diagonal).properties(
        width=360, height=360,
        title=alt.TitleParams("Do the two communities rate books alike?",
                              subtitle=[f"{len(frame):,} works with {min_ol_ratings}+ Open Library ratings. "
                                        f"Correlation {corr:.2f}; Open Library averages {mean_gap:+.2f} stars "
                                        "relative to Goodreads."])))
