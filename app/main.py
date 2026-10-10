"""One-screen recommender on Book 3's output: add books you've read, get five back with reasons.

The screen has four parts: a shelf of books the visitor has read (search and add), a choice of
recommendation source (the fused panel or one model), the five recommendations, and two charts that
explain the selected pick. Every question goes through `recommend.py`; this module only draws.

    python main.py      serves on $PORT (8000 if unset); the container sets 8080
"""

import os

from nicegui import ui

import recommend as rec

PICK_COLOUR = "#2a78d6"   # always a book we are recommending
YOURS_COLOUR = "#eb6834"  # always a book you have read


def bar(label, value, text, colour, height=14):
    """One labelled horizontal bar: `value` is the filled width in percent, `text` the figure on the right."""
    with ui.row().classes("w-full items-center gap-3 no-wrap"):
        ui.label(label).classes("w-44 text-right text-sm text-gray-600 shrink-0 truncate")
        with ui.element("div").classes("flex-grow rounded bg-gray-200").style(f"height: {height}px"):
            ui.element("div").classes("rounded").style(
                f"width: {value}%; height: {height}px; background: {colour}")
        ui.label(text).classes("w-14 text-right text-sm font-medium shrink-0")


def paired_bars(label, top_value, bottom_value):
    """Two stacked bars for one label: the pick (top, blue) against a shelf book (bottom, orange), in percent."""
    with ui.row().classes("w-full items-center gap-3 no-wrap"):
        ui.label(label).classes("w-44 text-right text-sm text-gray-600 shrink-0 truncate")
        with ui.column().classes("flex-grow gap-0.5"):
            for value, colour in ((top_value, PICK_COLOUR), (bottom_value, YOURS_COLOUR)):
                with ui.element("div").classes("w-full rounded bg-gray-200").style("height: 10px"):
                    ui.element("div").classes("rounded").style(
                        f"width: {value}%; height: 10px; background: {colour}")
        ui.label(f"{top_value} / {bottom_value}").classes(
            "w-16 text-right text-sm font-medium shrink-0")


@ui.page("/")
def index():
    """The page. State lives inside this function, so each visitor gets their own shelf."""
    ui.add_head_html('<meta name="viewport" content="width=device-width, initial-scale=1">')
    shelf: list[int] = []      # per visitor: catalogue positions of books read
    state = {"picks": [], "pick": None}

    with ui.column().classes("w-full max-w-3xl mx-auto p-4 gap-4"):
        ui.label("Book Recommender").classes("text-2xl font-bold")
        ui.label('"Readers also read…" · SIADS 593 Milestone I · Cain, Ehlen, Hook') \
          .classes("text-sm text-gray-500 -mt-3")

        with ui.card().classes("w-full"):
            ui.label("Books you've read").classes("font-semibold")
            ui.label("Search the catalogue and add a few. We recommend from these.") \
              .classes("text-sm text-gray-600")

            search = ui.input(placeholder="Search by title or author") \
                       .props("clearable dense outlined").classes("w-full")
            results = ui.column().classes("w-full gap-0")
            chips = ui.row().classes("w-full gap-2 mt-2")

            def add(book):
                """Put a search result on the shelf and clear the search box."""
                shelf.append(book)
                search.value = ""
                refresh()

            def remove(book):
                """Take a book off the shelf."""
                shelf.remove(book)
                refresh()

            def draw_results():
                """List up to eight catalogue matches for the search term, each with an Add link."""
                results.clear()
                term = search.value or ""
                if len(term.strip()) < 2:
                    return
                hits = rec.search(term, skip=shelf)
                with results:
                    if not hits:
                        ui.label("Nothing matched.").classes("text-sm text-gray-500 p-2")
                    for book in hits:
                        with ui.row().classes(
                            "w-full items-center justify-between gap-3 no-wrap "
                            "p-2 rounded hover:bg-gray-100 cursor-pointer"
                        ).on("click", lambda b=book: add(b)):
                            with ui.column().classes("gap-0 min-w-0"):
                                ui.label(rec.TITLES[book]).classes("text-sm font-medium truncate")
                                ui.label(rec.AUTHORS[book]).classes("text-xs text-gray-500 truncate")
                            ui.label("Add").classes("text-sm text-blue-600 shrink-0")

            def draw_chips():
                """Show the shelf as removable chips."""
                chips.clear()
                with chips:
                    if not shelf:
                        ui.label("Nothing added yet.").classes("text-sm text-gray-500")
                    for book in shelf:
                        with ui.row().classes(
                            "items-center gap-1 no-wrap px-2 py-1 rounded-full bg-orange-100"
                        ):
                            ui.label(rec.TITLES[book]).classes("text-sm")
                            ui.label("✕").classes("text-xs text-gray-500 cursor-pointer") \
                              .on("click", lambda b=book: remove(b))

            search.on_value_change(draw_results)

        with ui.card().classes("w-full"):
            with ui.row().classes("w-full items-end gap-3 no-wrap"):
                mode = ui.select(rec.MODES, value="fused", label="Recommendation source") \
                         .props("dense outlined").classes("flex-grow")
                ui.button("Recommend", on_click=lambda: run()).classes("shrink-0")

        output = ui.column().classes("w-full gap-4")

        ui.label("Recommendations are Book 3's output: each book's top five from the Jaccard-on-likes "
                 "and finished-together models, fused by reciprocal rank (k = 60), with same-author and "
                 "same-series books removed; several books on the shelf are fused the same way. "
                 "Catalogue: the 43,203 works liked by at least 10 training readers.") \
          .classes("text-xs text-gray-500")

        def run():
            """Ask for five recommendations for the current shelf and source, and select the first."""
            state["picks"] = rec.recommend(shelf, mode.value)
            state["pick"] = state["picks"][0] if state["picks"] else None
            draw_output()

        def choose(book):
            """Select a recommendation so the charts below explain it."""
            state["pick"] = book
            draw_output()

        def refresh():
            """Redraw everything that depends on the shelf."""
            draw_results()
            draw_chips()
            draw_output()

        def draw_output():
            """Draw the recommendations and, for the selected one, the two explanation charts."""
            output.clear()
            live = [b for b in state["picks"] if b not in shelf]   # a pick added to the shelf drops out
            if not live or not shelf:
                with output:
                    ui.label("Add a book or two, then press Recommend.") \
                      .classes("text-sm text-gray-500")
                return
            pick = state["pick"] if state["pick"] in live else live[0]

            with output:
                with ui.card().classes("w-full"):
                    ui.label("WE RECOMMEND").classes("text-xs tracking-widest text-gray-500")
                    ui.label("Pick one to see why.").classes("text-sm text-gray-600 mb-1")
                    for book in live:
                        with ui.row().classes(
                            "w-full items-center gap-3 no-wrap p-2 rounded cursor-pointer "
                            + ("bg-blue-50 border border-blue-300" if book == pick
                               else "hover:bg-gray-100")
                        ).on("click", lambda b=book: choose(b)):
                            with ui.column().classes("gap-0 min-w-0"):
                                ui.label(rec.TITLES[book]).classes("font-bold truncate")
                                ui.label(rec.AUTHORS[book]).classes("text-xs text-gray-500 truncate")

                with ui.card().classes("w-full"):
                    ui.label(f"Why {rec.TITLES[pick]}").classes("font-semibold")
                    ui.label("Where it ranks in the list for each book you've read. "
                             "A longer bar is a higher rank.") \
                      .classes("text-sm text-gray-600 mb-2")
                    # Rank 1 fills the bar, rank 5 fills a fifth of it.
                    for book, rank in rec.reasons(pick, shelf, mode.value):
                        bar(rec.TITLES[book], round(100 * (rec.TOP_N + 1 - rank) / rec.TOP_N),
                            f"#{rank} of {rec.TOP_N}", PICK_COLOUR)
                    ui.label("The book at the top is the one most responsible for this "
                             "recommendation.") \
                      .classes("text-sm text-gray-500 mt-2")

                with ui.card().classes("w-full"):
                    ui.label("Side by side with one of your books").classes("font-semibold")
                    ui.label("Share of each book's Goodreads genre shelvings, in percent.") \
                      .classes("text-sm text-gray-600 mb-2")
                    chooser = ui.select({b: rec.TITLES[b] for b in shelf}, value=shelf[0],
                                        label="Compare against") \
                                .props("dense outlined").classes("w-full mb-2")
                    chart = ui.column().classes("w-full gap-1")

                    def redraw():
                        """Draw the genre comparison against the shelf book chosen in the menu."""
                        chart.clear()
                        mine = rec.genre_shares(chooser.value)
                        theirs = rec.genre_shares(pick)
                        with chart:
                            with ui.row().classes("gap-4 text-sm text-gray-600 mb-1"):
                                for colour, book in ((PICK_COLOUR, pick), (YOURS_COLOUR, chooser.value)):
                                    with ui.row().classes("items-center gap-1.5 no-wrap"):
                                        ui.element("div").classes("rounded").style(
                                            f"width: 10px; height: 10px; background: {colour}")
                                        ui.label(rec.TITLES[book])
                            for i, genre in enumerate(rec.GENRES):
                                if theirs[i] or mine[i]:   # skip genres neither book is shelved under
                                    paired_bars(genre, theirs[i], mine[i])

                    chooser.on_value_change(redraw)
                    redraw()

        refresh()


ui.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), title="Book Recommender",
       reload=False, show=False)
