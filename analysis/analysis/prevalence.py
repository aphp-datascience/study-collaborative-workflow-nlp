import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis import helpers
from analysis.format_table import Clean
from analysis.helpers import Colors, Rename


def get_prevalence(
    side_to_side_metrics: pd.DataFrame,
    texts: pd.DataFrame,
):
    # df = pd.read_pickle("./data/side_to_side_metrics_ml.pickle")
    # texts = data.get_all("texts")

    note_id_to_cse = texts.query("patient_type=='inpatient'")[["note_id", "cse"]]

    # Adding CSE

    comorbs_list = list(side_to_side_metrics.label_name.unique())
    all_comorbs = []

    for comorb in comorbs_list:
        tmp = note_id_to_cse.copy()
        tmp["label_name"] = comorb
        all_comorbs.append(tmp)

    df = pd.concat(all_comorbs)
    df = helpers.add_overall(df)

    df = df.merge(
        side_to_side_metrics,
        on=["note_id", "label_name", "cse"],
        how="left",
    ).fillna(False)

    # Computing prevalence

    df = df.groupby(
        ["cse", "label_name"],
        as_index=False,
    ).agg(
        N_Gold=("gold", "sum"),
        N_NLP=("nlp", "sum"),
        N_Claim=("claim", "sum"),
        total=("gold", "size"),
    )

    # Pivot

    df = pd.wide_to_long(
        df,
        stubnames="N",
        sep="_",
        i=["cse", "label_name"],
        j="method",
        suffix=".*",
    )

    # Final prevalence

    df["Prevalence (%)"] = (100 * df["N"] / df["total"]).round(1)

    # Formatting (sort, rename, etc.)

    df = Clean.run(df)
    df.reset_index(inplace=True)
    df["label_name"] = df["label_name"].apply(Rename.run)

    return df


def plot_prevalence(
    prevalences: pd.DataFrame,
):
    plt.close()

    HEIGHT = 9
    RATIO = 1.4142

    fig, axes = plt.subplots(4, 1, figsize=(RATIO * HEIGHT, HEIGHT), sharex=True)
    axes = axes.flatten()

    ax_label = "abcd"
    n_stays = [150, 50, 50, 50]

    for ax, ax_label, n_stay, (cse, df) in zip(
        axes, ax_label, n_stays, prevalences.groupby("cse")
    ):
        g = sns.barplot(
            data=df,
            x="label_name",
            y="Prevalence (%)",
            hue="method",
            ax=ax,
            width=0.6,
            log=False,
            palette=Colors,
        )

        ax.set_ylabel("", fontsize=13)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right")
        _ = ax.set(
            xlabel="",
            ylabel="Prevalence",
        )

        ax.tick_params(axis="both", which="major", labelsize=12)

        ax.set_title("")

        g.set_title(
            f"({ax_label}) " + Rename.run(cse) + f" cohort (N={n_stay})",
            fontweight="bold" if (cse == "Overall") else None,
        )
        ax.get_legend().remove()

        ax.set_yticks(
            [0, 5, 15, 20, 30, 40, 50, 75], [None, 5, 15, None, 30, None, 50, 75]
        )

        if cse == "Overall":
            ax.set_facecolor(3 * [0.85])

        ax.grid(axis="y", which="major")
        ax.set_axisbelow(True)

        hatches = "\\\\\\"
        gold_bars = ax.containers[0]
        for bar in gold_bars:
            bar.set_hatch(hatches)
            bar.set_edgecolor("black")
            bar.set_zorder(100)

        handles, labels = ax.get_legend_handles_labels()
        # labels = [r"$\bf{Chart}$" + " " + r"$\bf{review}$"] + labels[1:]
        labels = [
            r"$\bf{Chart}$" + " " + r"$\bf{review}$",
            r"$\bf{NLP-ML-CLINICAL}$" + " pipeline",
            r"$\bf{CLAIM}$" + " pipeline",
        ]
    fig.legend(handles, labels, fancybox=True, framealpha=1, fontsize=12)

    return fig
