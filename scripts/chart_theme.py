"""One palette and one set of design rules for every chart in Books 1–3.

Every chart module imports from here, so a change to a colour or a type size reaches all
notebooks at once and the charts read as one set.

Colour roles
    Categorical: the three recommenders always take the same slots, in the same order —
    directional = BLUE, Jaccard on likes = ORANGE, finished together = AQUA. Anything that is
    context (a baseline, a reference line) is GREY or INK, never a fourth hue.
    Sequential: BLUE_RAMP, light to dark, for "how many" (the evidence explorer's cells).
    Highlight: ORANGE for the single thing the reader asked for (a title search hit).
    Fused recommenders: INK diamonds, filled for the agreed setting and hollow for a variant.

Marks
    Circles for single models, diamonds for fused ones, grey bars for ranges, dashed INK2 rules
    for reference values, 2 px lines, direct labels where they fit and a legend otherwise.

Type and chrome
    System sans; title 14, subtitle 11.5, axis labels 11, axis titles 12 (not bold); hairline
    grid; legends at the bottom.
"""

from __future__ import annotations

import altair as alt

# ink and chrome
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

# categorical slots (validated colour-blind safe as a set) and context grey
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9892"
ORANGE_INK = "#b84a1e"          # darker orange for text placed beside orange marks

# sequential ramp for counts: light = few, dark = many
BLUE_RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# the three recommenders, in the order every chart uses
MODEL_ORDER = ["Directional (likes)", "Jaccard (likes)", "Finished together (shelves)"]
MODEL_COLOURS = [BLUE, ORANGE, AQUA]


def model_scale(names) -> alt.Scale:
    """Colour scale for a set of recommender names: models keep their slots, everything else is grey."""
    domain = [m for m in MODEL_ORDER if m in names] + [n for n in names if n not in MODEL_ORDER]
    colours = [MODEL_COLOURS[MODEL_ORDER.index(n)] if n in MODEL_ORDER else GREY for n in domain]
    return alt.Scale(domain=domain, range=colours)


def theme(chart):
    """Apply the shared chrome. Call once, on the top-level chart (Vega-Lite allows one configuration)."""
    return (chart.configure(background=SURFACE, font=FONT)
            .configure_view(stroke=None)
            .configure_axis(gridColor=GRID, domainColor=AXIS, tickColor=AXIS, labelColor=INK2, titleColor=INK2,
                            labelFontSize=11, titleFontSize=12, titleFontWeight="normal", titlePadding=8)
            .configure_title(color=INK, subtitleColor=INK2, fontSize=14, subtitleFontSize=11.5, anchor="start",
                             offset=10)
            .configure_legend(labelColor=INK2, titleColor=INK2, labelFontSize=11, titleFontSize=11))
