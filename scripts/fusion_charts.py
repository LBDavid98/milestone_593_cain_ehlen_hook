"""The one chart in Book 3: where fusion helps and where it averages, one line per metric.

    fusion_range_chart(results)   grey bar = range of the three single models on each metric,
                                  dots = the single models, filled diamond = the agreed fusion
                                  (top 5 from each model), hollow diamond = the two-Jaccard variant.
                                  A diamond beyond the bar is a gain; inside it, an average.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

from chart_theme import FONT, GRID, INK, MODEL_COLOURS, MODEL_ORDER, SURFACE, theme as _theme

SINGLE_MODELS = MODEL_ORDER
SINGLE_COLOURS = MODEL_COLOURS
METRICS = [("precision@5", "Precision at 5"), ("precision@10", "Precision at 10"), ("recall@10", "Recall at 10"),
           ("hit@10", "Hit rate at 10"), ("ndcg@10", "NDCG at 10"),
           ("Goodreads hit@10", "Agreement with Goodreads lists")]


def fusion_range_chart(results: pd.DataFrame, fused: str = "Fused (top 5 each)",
                       fused_two: str = "Fused, two Jaccard models only (top 5 each)",
                       metrics=METRICS) -> alt.LayerChart:
    """One row per metric. The grey bar spans the three single models' scores, each marked with a
    dot in its model colour; the filled diamond is the agreed fused recommender and the hollow
    diamond the two-Jaccard variant. Reading: a diamond to the right of the bar is a gain from
    fusion, a diamond inside the bar is an average of the models it combines.

    results: the test table with recommenders as the index (Book 3's `book3_test_results.pkl`).
    """
    rows = []
    for column, label in metrics:
        for model in SINGLE_MODELS + [fused, fused_two]:
            kind = "fused (top 5 from each model)" if model == fused else (
                "fused (two Jaccard models only)" if model == fused_two else "single model")
            rows.append({"metric": label, "model": model, "value": float(results.loc[model, column]), "kind": kind})
    frame = pd.DataFrame(rows)
    order = [label for _, label in metrics]
    singles = frame[frame["kind"] == "single model"]
    fused_rows = frame[frame["kind"] != "single model"]
    span = singles.groupby("metric")["value"].agg(["min", "max"]).reset_index()
    kinds = ["fused (top 5 from each model)", "fused (two Jaccard models only)"]

    y = alt.Y("metric:N", sort=order, title=None, axis=alt.Axis(labelLimit=240, labelFontSize=12))
    x = alt.X("value:Q", title="Score on test readers", axis=alt.Axis(format=".0%"))
    band = alt.Chart(span).mark_rule(color=GRID, strokeWidth=10).encode(y=y, x="min:Q", x2="max:Q")
    dots = alt.Chart(singles).mark_circle(size=110, stroke=SURFACE, strokeWidth=1.5).encode(
        y=y, x=x,
        color=alt.Color("model:N", scale=alt.Scale(domain=SINGLE_MODELS, range=SINGLE_COLOURS),
                        legend=alt.Legend(title="Single models", orient="bottom", columns=3)),
        tooltip=[alt.Tooltip("model:N", title="Recommender"), alt.Tooltip("value:Q", format=".3f")])
    diamonds = alt.Chart(fused_rows).mark_point(shape="diamond", size=170, strokeWidth=2.2, color=INK).encode(
        y=y, x=x,
        fill=alt.Fill("kind:N", scale=alt.Scale(domain=kinds, range=[INK, SURFACE]),
                      legend=alt.Legend(title="Fused", orient="bottom", symbolType="diamond", symbolStrokeColor=INK,
                                        symbolStrokeWidth=2, symbolSize=170, labelLimit=320)),
        tooltip=[alt.Tooltip("model:N", title="Recommender"), alt.Tooltip("value:Q", format=".3f")])
    labels = alt.Chart(fused_rows[fused_rows["kind"] == kinds[0]]).mark_text(
        align="left", dx=10, dy=-11, fontSize=10.5, color=INK, font=FONT).encode(
        y=y, x=x, text=alt.Text("value:Q", format=".1%"))
    return _theme(
        alt.layer(band, dots, diamonds, labels).resolve_scale(color="independent", fill="independent")
        .properties(width=620, height=270, title=alt.TitleParams(
            "Where fusion helps and where it averages",
            subtitle=["Grey bar: the range of the three single models. Filled diamond: fused, top 5 from each model.",
                      "Hollow diamond: fused, two Jaccard models only. A diamond beyond the bar is a gain; inside it, "
                      "an average."])))
