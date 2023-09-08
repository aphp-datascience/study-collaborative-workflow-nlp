import os

import pandas as pd

from analysis import data
from analysis.mentionned_charlson import plot_charlson_comparison


def main(
    notes: pd.DataFrame,
    visits: pd.DataFrame,
):
    fig, results = plot_charlson_comparison(notes, visits)

    data.save(
        fig,
        "mentionned_charlson_histogram",
        "Mentionned charlson histogram",
        bbox_inches="tight",
        export=True,
    )

    data.save(
        results,
        "mentionned_charlson_stats",
        "Mentionned charlson stats",
        export=True,
    )


if __name__ == "__main__":
    os.environ["CSE"] = "Overall"
    os.environ["MODEL"] = "eds"

    notes = data.get("mentionned_charlson_ml")
    visits = data.get_all_raw("mentionned_charlson")

    main(
        notes=notes,
        visits=visits,
    )
