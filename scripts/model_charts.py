"""Charts for Book 2's evaluation section. Each one answers a single question about the models.

    accuracy_vs_popularity(results, seed_popularity)   Are the accurate models accurate because they
                                                       recommend bestsellers?
    concentration_curves(counts)                       How many different books does each model
                                                       actually recommend?
    filter_tradeoff(results_without, results_with)     What does removing same-author and same-series
                                                       books do to the two measures of quality?
    recommendation_inspector(examples)                 For a chosen book, what do the three models
                                                       return, and how popular is each pick?

The functions take the frames the notebook already has (the evaluation table, recommendation
counts, an examples table) and return Altair charts. Colours: the three models use the project's
three categorical slots; baselines are grey so they read as context.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from chart_theme import (FONT, GREY, INK, INK2, MODEL_COLOURS, MODEL_ORDER, SURFACE, model_scale as _colour_scale,
                         theme as _theme)


def accuracy_vs_popularity(results: pd.DataFrame, seed_popularity: float) -> alt.LayerChart:
    """One dot per recommender: how accurate it is (precision at 5, y) against how mainstream its
    recommendations are (median training readers of the works it recommends, x, log scale).
    The dashed line is the median popularity of the works held-out readers actually liked, so a
    dot far to its right recommends books much more widely read than the reader's own taste."""
    frame = results.reset_index().rename(columns={"index": "model"})
    frame = frame[["model", "precision@5", "median popularity (recs)", "coverage"]].copy()
    frame["kind"] = np.where(frame["model"].isin(MODEL_ORDER), "model", "baseline")
    frame = frame[frame["median popularity (recs)"] > 0]

    x = alt.X("median popularity (recs):Q", scale=alt.Scale(type="log"),
              axis=alt.Axis(format="~s",
                            title="Median popularity of the recommended works (training readers, log scale)"))
    y = alt.Y("precision@5:Q", title="Precision at 5 on held-out readers", axis=alt.Axis(format=".0%"))
    dots = alt.Chart(frame).mark_circle(size=170, stroke=SURFACE, strokeWidth=2).encode(
        x=x, y=y, color=alt.Color("model:N", scale=_colour_scale(list(frame["model"])), legend=None),
        tooltip=[alt.Tooltip("model:N", title="Recommender"), alt.Tooltip("precision@5:Q", format=".3f"),
                 alt.Tooltip("median popularity (recs):Q", title="Median popularity", format=","),
                 alt.Tooltip("coverage:Q", title="Distinct works recommended", format=",")])
    labels = alt.Chart(frame).mark_text(align="left", dx=11, dy=-1, fontSize=11.5, color=INK, font=FONT).encode(
        x=x, y=y, text="model:N")
    marker = pd.DataFrame({"x": [seed_popularity]})
    seeds_rule = alt.Chart(marker).mark_rule(color=INK2, strokeDash=[4, 4]).encode(x="x:Q")
    seeds_label = alt.Chart(marker).mark_text(
        align="right", dx=-6, dy=0, fontSize=11, color=INK2, font=FONT).encode(
        x="x:Q", y=alt.value(306), text=alt.value("median popularity of the books readers liked"))
    return _theme(alt.layer(dots, labels, seeds_rule, seeds_label).properties(
        width=620, height=320,
        title=alt.TitleParams("Accuracy against how mainstream the recommendations are",
                              subtitle="Up and left is better: accurate without defaulting to bestsellers. "
                                       "The directional score sits beside the most-popular baseline.")))


def concentration_curves(counts: dict[str, pd.Series], points: int = 200) -> alt.LayerChart:
    """Cumulative share of all recommendations made by each recommender, against the number of
    distinct works ranked by how often they were recommended (log x). A curve that reaches 100%
    after a few hundred works is a recommender that shows everyone the same books. The legend
    carries each recommender's total number of distinct works."""
    rows, labels = [], {}
    for name, series in counts.items():
        values = np.sort(series.to_numpy())[::-1]
        cumulative = np.cumsum(values) / values.sum()
        ranks = np.unique(np.round(np.geomspace(1, len(values), points)).astype(int))
        labels[name] = f"{name}: {len(values):,} works"
        rows += [{"model": labels[name], "works": int(r), "share": float(cumulative[r - 1])} for r in ranks]
    frame = pd.DataFrame(rows)
    order = [m for m in MODEL_ORDER if m in counts] + [n for n in counts if n not in MODEL_ORDER]
    scale = alt.Scale(domain=[labels[n] for n in order],
                      range=[MODEL_COLOURS[MODEL_ORDER.index(n)] if n in MODEL_ORDER else GREY for n in order])

    x = alt.X("works:Q", scale=alt.Scale(type="log"),
              axis=alt.Axis(format="~s", title="Distinct works, most-recommended first (log scale)"))
    y = alt.Y("share:Q", axis=alt.Axis(format="%", title="Share of all recommendations made"))
    lines = alt.Chart(frame).mark_line(strokeWidth=2).encode(
        x=x, y=y,
        color=alt.Color("model:N", scale=scale, sort=list(scale.domain),
                        legend=alt.Legend(title=None, orient="right", symbolType="stroke", symbolStrokeWidth=3,
                                          labelLimit=320)),
        strokeDash=alt.StrokeDash("model:N", sort=list(scale.domain), legend=None,
                                  scale=alt.Scale(domain=list(scale.domain),
                                                  range=[[1, 0]] * 3 + [[6, 4], [2, 3]][:len(order) - 3])),
        tooltip=[alt.Tooltip("model:N", title="Recommender"), alt.Tooltip("works:Q", format=","),
                 alt.Tooltip("share:Q", format=".0%")])
    return _theme(alt.layer(lines).properties(
        width=560, height=300,
        title=alt.TitleParams("How many different books each recommender actually shows",
                              subtitle="The directional score and the popularity baselines reach 100% within a few "
                                       "hundred works; the two Jaccard models draw on thousands.")))


def filter_tradeoff(results_without: pd.DataFrame, results_with: pd.DataFrame) -> alt.LayerChart:
    """For each model, an arrow from its scores without the same-author/same-series filter to its
    scores with it, in the plane of the two quality measures: precision on held-out readers (x) and
    agreement with Goodreads' own lists (y). The arrows point the same way for every model."""
    rows = []
    for model in MODEL_ORDER:
        off = results_without.loc[model]
        on = results_with.loc[[i for i in results_with.index if i.startswith(model)][0]]
        rows.append({"model": model, "x": off["precision@5"], "y": off["Goodreads hit@10"],
                     "x2": on["precision@5"], "y2": on["Goodreads hit@10"]})
    frame = pd.DataFrame(rows)
    scale = _colour_scale(MODEL_ORDER)
    x = alt.X("x:Q", title="Precision at 5 on held-out readers", axis=alt.Axis(format=".0%"),
              scale=alt.Scale(domain=[0, 0.25]))
    y = alt.Y("y:Q", title="Share of works whose top 10 overlaps Goodreads' list", axis=alt.Axis(format=".0%"),
              scale=alt.Scale(domain=[0, 0.6]))
    arrows = alt.Chart(frame).mark_rule(strokeWidth=2).encode(
        x=x, y=y, x2="x2:Q", y2="y2:Q",
        color=alt.Color("model:N", scale=scale, legend=alt.Legend(title=None, orient="bottom")))
    start = alt.Chart(frame).mark_circle(size=90, opacity=0.5).encode(
        x=x, y=y, color=alt.Color("model:N", scale=scale, legend=None))
    end = alt.Chart(frame).mark_point(shape="triangle", size=120, filled=True, angle=0).encode(
        x="x2:Q", y="y2:Q", color=alt.Color("model:N", scale=scale, legend=None))
    start_label = alt.Chart(frame.iloc[[1]]).mark_text(align="left", dx=10, dy=12, fontSize=11, color=INK2,
                                                       font=FONT).encode(
        x=x, y=y, text=alt.value("without filter"))
    end_label = alt.Chart(frame.iloc[[1]]).mark_text(align="left", dx=10, dy=-6, fontSize=11, color=INK2,
                                                     font=FONT).encode(
        x="x2:Q", y="y2:Q", text=alt.value("with filter"))
    return _theme(alt.layer(arrows, start, end, start_label, end_label).properties(
        width=480, height=320,
        title=alt.TitleParams("Removing same-author and same-series books: the two quality measures disagree",
                              subtitle="Held-out readers often liked the sequel, so precision falls; Goodreads' lists "
                                       "exclude same-author titles, so agreement with them rises.")))


def recommendation_inspector(examples: pd.DataFrame, default_seed: str) -> alt.FacetChart:
    """Pick a seed book from the dropdown; one column per recommender shows its five picks (rank
    order, top to bottom) with a bar for how many training readers have each pick. The directional
    score's bars are long for every seed: it returns bestsellers whatever the book.

    examples: columns seed (title), model, rank (1-5), title, popularity.
    """
    seeds = sorted(examples["seed"].unique().tolist())
    picker = alt.selection_point(fields=["seed"], value=default_seed,
                                 bind=alt.binding_select(options=seeds, name="Seed book  "))
    bars = (
        alt.Chart(examples).transform_filter(picker)
        .mark_bar(cornerRadiusEnd=3, height=16, opacity=0.55)
        .encode(x=alt.X("popularity:Q", axis=alt.Axis(format="~s", title="Training readers who have the work"),
                        scale=alt.Scale(nice=True)),
                y=alt.Y("rank:O", title=None, axis=alt.Axis(labels=False, ticks=False)),
                color=alt.Color("model:N", scale=_colour_scale(MODEL_ORDER), legend=None),
                tooltip=[alt.Tooltip("title:N", title="Recommendation"),
                         alt.Tooltip("popularity:Q", format=",", title="Readers")])
    )
    text = (
        alt.Chart(examples).transform_filter(picker)
        .mark_text(align="left", dx=4, fontSize=11, color=INK, font=FONT, limit=250)
        .encode(x=alt.value(0), y=alt.Y("rank:O"), text=alt.Text("title:N"))
    )
    return _theme(
        alt.layer(bars, text).properties(width=250, height=130)
        .facet(column=alt.Column("model:N", sort=MODEL_ORDER, title=None,
                                 header=alt.Header(labelFontSize=12, labelColor=INK)))
        .add_params(picker)
        .properties(title=alt.TitleParams("Pick a book and compare the five recommendations",
                                          subtitle="Bars show how widely read each recommendation is. Same-author and "
                                                   "same-series works are filtered out, as on the dashboard."))
    )


def two_pane(counts: dict[str, pd.Series], examples: pd.DataFrame, default_seed: str) -> alt.HConcatChart:
    """One panel with two views: the concentration curves on the left, and on the right the
    recommendation inspector as three stacked rows (one per model) driven by a single dropdown.
    Both views are built unconfigured and themed once here, because Vega-Lite allows one
    configuration per top-level chart."""
    seeds = sorted(examples["seed"].unique().tolist())
    picker = alt.selection_point(fields=["seed"], value=default_seed,
                                 bind=alt.binding_select(options=seeds, name="Seed book  "))
    x_max = float(examples["popularity"].max()) * 1.05

    def row(model, first):
        frame = examples[examples["model"] == model]
        base = alt.Chart(frame).transform_filter(picker)
        bars = base.mark_bar(cornerRadiusEnd=3, height=14, opacity=0.55,
                             color=MODEL_COLOURS[MODEL_ORDER.index(model)]).encode(
            x=alt.X("popularity:Q", scale=alt.Scale(domain=[0, x_max]),
                    axis=alt.Axis(format="~s", labels=model == MODEL_ORDER[-1],
                                  title="Training readers who have the work" if model == MODEL_ORDER[-1] else None)),
            y=alt.Y("rank:O", title=None, axis=alt.Axis(labels=False, ticks=False)),
            tooltip=[alt.Tooltip("title:N", title="Recommendation"),
                     alt.Tooltip("popularity:Q", format=",", title="Readers")])
        if first:
            bars = bars.add_params(picker)
        text = base.mark_text(align="left", dx=4, fontSize=10.5, color=INK, font=FONT, limit=300).encode(
            x=alt.value(0), y=alt.Y("rank:O"), text=alt.Text("title:N"))
        return alt.layer(bars, text).properties(width=330, height=100, title=alt.TitleParams(model, fontSize=12))

    inspector = alt.vconcat(*[row(m, i == 0) for i, m in enumerate(MODEL_ORDER)], spacing=10).properties(
        title=alt.TitleParams("Five recommendations for the chosen book",
                              subtitle=["Bars: how widely read each pick is.",
                                        "Same-author and same-series works removed, as on the dashboard."]))

    rows, labels = [], {}
    for name, series in counts.items():
        values = np.sort(series.to_numpy())[::-1]
        cumulative = np.cumsum(values) / values.sum()
        ranks = np.unique(np.round(np.geomspace(1, len(values), 200)).astype(int))
        labels[name] = f"{name}: {len(values):,} works"
        rows += [{"model": labels[name], "works": int(r), "share": float(cumulative[r - 1])} for r in ranks]
    frame = pd.DataFrame(rows)
    order = [m for m in MODEL_ORDER if m in counts] + [n for n in counts if n not in MODEL_ORDER]
    domain = [labels[n] for n in order]
    colours = [MODEL_COLOURS[MODEL_ORDER.index(n)] if n in MODEL_ORDER else GREY for n in order]
    scale = alt.Scale(domain=domain, range=colours)
    concentration = alt.Chart(frame).mark_line(strokeWidth=2).encode(
        x=alt.X("works:Q", scale=alt.Scale(type="log"),
                axis=alt.Axis(format="~s", title="Distinct works, most-recommended first (log scale)")),
        y=alt.Y("share:Q", axis=alt.Axis(format="%", title="Share of all recommendations made")),
        color=alt.Color("model:N", scale=scale, sort=domain,
                        legend=alt.Legend(title=None, orient="bottom", columns=1, symbolType="stroke",
                                          symbolStrokeWidth=3, labelLimit=320)),
        strokeDash=alt.StrokeDash("model:N", sort=domain, legend=None,
                                  scale=alt.Scale(domain=domain,
                                                  range=[[1, 0]] * 3 + [[6, 4], [2, 3]][:len(order) - 3])),
        tooltip=[alt.Tooltip("model:N", title="Recommender"), alt.Tooltip("works:Q", format=","),
                 alt.Tooltip("share:Q", format=".0%")],
    ).properties(width=420, height=300, title=alt.TitleParams(
        "How many different books each recommender shows",
        subtitle=["Cumulative share of all test recommendations.",
                  "Steep curves show everyone the same few books."]))

    return _theme(alt.hconcat(concentration, inspector, spacing=40).resolve_scale(color="independent"))
