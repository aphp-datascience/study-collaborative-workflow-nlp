import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis import data
from analysis.helpers import Colors, Rename, add_overall
from analysis.metrics import prepare_icd10_metrics, prepare_note_metrics


def plot_charlson_comparison(
    notes: pd.DataFrame,
    visits: pd.DataFrame,
):
    # notes = pd.read_pickle(DATA_DIR / "mentionned_charlson_ml.pickle")
    # visits = data.get_all("mentionned_charlson")

    notes["patient_type"] = "inpatient"

    # note -> visit_occurrence
    note_visit_mapping = visits[
        ["note_id", "visit_occurrence_id", "cse"]
    ].drop_duplicates(subset=["visit_occurrence_id", "note_id"])

    # adding visit_occurrence_id value
    notes = notes.merge(
        note_visit_mapping,
        on=["note_id", "visit_occurrence_id"],
        how="inner",
    )

    # aggregating labels at note level
    ml = prepare_note_metrics(
        notes,
        label_type="ml",
        group_columns=["label_name", "cse"],
        with_aggregated=False,
    )

    # we can have the same note_id in different CSE
    visits = visits.drop_duplicates(subset=["visit_occurrence_id", "note_id"])

    # aggregating labels at note level
    icd10 = prepare_icd10_metrics(
        visits.drop("note_id", axis=1),
        visits[["note_id", "visit_occurrence_id"]].copy(),
        with_aggregated=False,
    )

    results = pd.concat([ml, icd10])

    # get Charlson weights per comorbidity
    weights = data.get_charlson_weights(revision="quan")

    results = results.merge(
        weights,
        on=["label_name", "label_value"],
        how="left",
        validate="many_to_one",
    )

    # grouping to merge together leukemia, lymphoma and cancer
    results = (
        results.drop("label_name", axis=1)
        .rename(columns={"label_name_updated": "label_name"})
        .groupby(["cse", "note_id", "source", "label_name"])
        .agg(ecci_weight=("ecci_weight", "max"))
    )

    # computing Charlson
    results = (
        results.groupby(["cse", "note_id", "source"])
        .agg(ecci=("ecci_weight", "sum"))
        .reset_index()
    )

    # adding mentionned Charlson
    mentionned = visits[["mentionned_charlson", "note_id", "cse"]]
    mentionned["source"] = "mentionned"
    mentionned = mentionned.rename(columns=dict(mentionned_charlson="ecci"))
    mentionned = add_overall(mentionned)

    results = pd.concat([results, mentionned])

    results_wide = (
        results.pivot(
            index=["cse", "note_id"],
            columns="source",
            values="ecci",
        )
        .fillna(0)
        .reset_index()
    )

    results_wide["claim_vs_nlp"] = -(results_wide["icd10"] - results_wide["ml"])
    results_wide["mentionned_vs_nlp"] = -(
        results_wide["mentionned"] - results_wide["ml"]
    )

    # plotting histogram

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=False)
    axes = axes.flatten()

    ax_label = "abcd"

    for ax, ax_label, (cse, df) in zip(axes, ax_label, results_wide.groupby("cse")):
        N = len(df)

        stats = dict(
            claim=dict(
                mean=df["claim_vs_nlp"].mean(),
                std=df["claim_vs_nlp"].std(),
                q1=df["claim_vs_nlp"].quantile(q=0.25),
                q3=df["claim_vs_nlp"].quantile(q=0.75),
                median=df["claim_vs_nlp"].median(),
            ),
            mentionned=dict(
                mean=df["mentionned_vs_nlp"].mean(),
                std=df["mentionned_vs_nlp"].std(),
                q1=df["mentionned_vs_nlp"].quantile(q=0.25),
                q3=df["mentionned_vs_nlp"].quantile(q=0.75),
                median=df["mentionned_vs_nlp"].median(),
            ),
            gold=dict(
                mean=df["mentionned"].mean(),
                median=df["mentionned"].median(),
            ),
        )

        df["mentionned"].mean()

        handles = []

        # Claim
        sns.histplot(
            df["claim_vs_nlp"],
            edgecolor="black",
            discrete=True,
            legend=True,
            stat="percent",
            ax=ax,
            color=Colors["Claim"],
        )

        # turn the histogram upside down
        for patch in ax.patches:
            patch.set_height(-patch.get_height())

        handles.append(ax.get_children()[0])

        # NLP
        sns.histplot(
            df["mentionned_vs_nlp"],
            edgecolor="black",
            stat="percent",
            discrete=True,
            legend=True,
            ax=ax,
            color=Colors["Mentionned"],
        )

        handles.append(ax.get_children()[-11])

        # reversing to display NLP first in legend
        handles = handles[::-1]

        ax.set_xticks(np.arange(-15, 15, 5))
        ax.set_yticks(np.arange(0, 35, 5))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _ = ax.set(xlabel=r"$\Delta(CCI)$", ylabel="%")
        ax.xaxis.set_label_coords(0.9, 0.58)

        pos_ticks = np.array([t for t in ax.get_yticks() if t > 0])
        ticks = np.concatenate([-pos_ticks[::-1], [0], pos_ticks])

        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{abs(t)}" for t in ticks])
        ax.spines["bottom"].set_position("zero")

        ax.set(xlim=[-15, 15])

        handles.append(
            ax.vlines(
                x=stats["claim"]["median"],
                color=Colors["Claim"],
                label="median",
                ymin=-35,
                ymax=0,  # 0.49,
                linewidth=2,
            )
        )
        handles.append(
            ax.vlines(
                x=stats["mentionned"]["median"],
                color=Colors["Mentionned"],
                label="median",
                ymin=0,  # 0.505,
                ymax=35,
                linewidth=2,
            )
        )
        ax.add_patch(
            plt.Rectangle(
                (
                    stats["mentionned"]["q1"],
                    0,
                ),  # (x,y) of bottom left corner
                stats["mentionned"]["q3"] - stats["mentionned"]["q1"],  # width
                35,  # height
                alpha=0.3,
                color=Colors["Mentionned"],
            )
        )
        ax.add_patch(
            plt.Rectangle(
                (
                    stats["claim"]["q1"],
                    0,
                ),  # (x,y) of bottom left corner
                stats["claim"]["q3"] - stats["claim"]["q1"],  # width
                -35,
                alpha=0.3,
                color=Colors["Claim"],
            )
        )

        # Only 2 labels for legend:
        handles = handles[:2]

        labels = [
            r"$\bf{Reported}$" + " CCI",
            r"$\bf{CLAIM}$" + " pipeline",
        ]

        ax.set_title(
            f"({ax_label}) "
            + Rename.run(cse)
            + f" cohort (N={N}, median CCI={int(stats['gold']['median'])})",
            fontweight="bold" if (cse == "Overall") else None,
        )

        if cse == "Overall":
            ax.set_facecolor(3 * [0.85])

    # ax.legend(handles, labels, loc="upper left")
    fig.tight_layout()
    # fig.subplots_adjust(bottom=0.3, wspace=0.33)
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0, -0.05),
        fancybox=False,
        shadow=False,
        ncol=2,
    )

    return fig, results_wide
