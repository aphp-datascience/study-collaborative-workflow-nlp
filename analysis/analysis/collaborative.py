from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from analysis import data
from analysis.bootstrap import sample_rows
from analysis.config import CSES, RARE_COMORBS
from analysis.helpers import Rename, remove_aggregated_label


def compute_metrics(TP=0, FP=0, FN=0):
    """
    Compute metric from TP, FP and FN count
    """
    tp_fp = (TP + FP) or np.inf
    tp_fn = (TP + FN) or np.inf

    PPV = TP / tp_fp
    Sensitivity = TP / tp_fn
    F1 = 0 if ((tp_fp is np.inf) or (tp_fn is np.inf)) else 2 * TP / (2 * TP + FP + FN)
    Support = TP + FN

    return dict(
        PPV=100 * PPV,
        Sensitivity=100 * Sensitivity,
        F1=100 * F1,
        Support=Support,
    )


def get_ci(r: pd.Series):
    """
    Get the 95% CI from a Series
    """
    return f"({r.quantile(0.025):.1f} - {r.quantile(0.975):.1f})"


def compute_all_metrics(stats):
    results = dict(label_name=[], pred=[])
    for _, row in stats.iterrows():
        results["pred"].extend(int(row.TP) * ["TP"])
        results["pred"].extend(int(row.FP) * ["FP"])
        results["pred"].extend(int(row.FN) * ["FN"])
        results["label_name"].extend(
            int(row.TP + row.FP + row.FN) * [row["label_name"]]
        )
    results = pd.DataFrame(results)
    results.value_counts()

    micro = compute_metrics(**results.groupby("pred").size().to_dict())

    n_labels = 0
    support = 0
    macro = defaultdict(lambda: 0)
    weighted = defaultdict(lambda: 0)

    df = results.groupby(["label_name", "pred"]).size().to_frame().reset_index()
    df = df.pivot(index="label_name", columns="pred", values=0).fillna(0).reset_index()
    df = df[~df.label_name.isin(["Total"] + RARE_COMORBS)]

    for comorb in df.to_dict(orient="records"):
        comorb.pop("label_name")
        comorb_metrics = compute_metrics(**comorb)

        n_labels += 1 if comorb_metrics["Support"] else 0
        support += comorb_metrics["Support"]

        for metric, value in comorb_metrics.items():
            macro[metric] += value
            weighted[metric] += value * comorb_metrics["Support"]

    for metric, value in macro.items():
        macro[metric] = value / n_labels

    for metric, value in weighted.items():
        weighted[metric] = value / support

    all_metrics = dict(macro=macro, micro=micro, weighted=weighted)

    return pd.DataFrame(all_metrics)


def bootstrap(stats, N, stratify=None):
    all_ci = []
    true_stats = stats.copy()

    cols = list(
        set(["label_name", "TP", "FP", "FN", "TN", "Support"]) & set(stats.columns)
    )

    if stratify is not None:
        true_stats = true_stats[cols].groupby("label_name", as_index=False).sum()
    f1 = compute_all_metrics(true_stats)

    for i in tqdm(range(N)):
        # sample = stats.sample(replace=True, frac=1)
        if stratify is not None:
            samples = []
            for _, stratified_df in stats.groupby(stratify):
                assert stratified_df.label_name.nunique() == len(
                    stratified_df
                ), "We should have one row per comorbidity"
                samples.append(stratified_df.apply(sample_rows, axis=1))
            sample = pd.concat(samples)
            sample = sample[cols].groupby("label_name", as_index=False).sum()
        else:
            assert stats.label_name.nunique() == len(
                stats
            ), "We should have one row per comorbidity"
            sample = stats.apply(sample_rows, axis=1)
        all_ci.append(compute_all_metrics(sample))

    all_ci = pd.concat(all_ci)
    all_ci = all_ci.groupby(all_ci.index).agg(get_ci)
    all_ci = pd.melt(
        all_ci.reset_index(), id_vars="index", var_name="method", value_name="CI"
    )

    metrics = pd.melt(
        f1.reset_index(), id_vars="index", var_name="method", value_name="value"
    )

    metrics = metrics.merge(all_ci, on=["index", "method"]).rename(
        columns=dict(index="metric")
    )

    return metrics


def get_metrics_with_ci(mode: str, model: Optional[str] = None):
    filename = (
        "bs_note_metrics_ml_all_patients"
        if (mode == "full_on_notes")
        else "bs_entity_metrics_ml_on_gold"
    )

    ALL_CSES = ["Overall"] + CSES
    all_metrics = []

    for train_cse in ALL_CSES:
        df = data.get(filename, cse=train_cse, model=model).reset_index()
        metrics = (
            df.query("label_name=='Total'")
            .reset_index()
            .melt(["label_value", "cse"], ["F1", "PPV", "label_value"])
        )

        metrics["CI"] = metrics["value"].str.extract(r"(\(.*\))")

        metrics["value"] = metrics.value.str.split(" ").str[0].astype(float)
        metrics = metrics.rename(
            columns=dict(cse="validated_on", label_value="method", variable="metric")
        )

        metrics = metrics.replace(
            {
                "Macro average": "macro",
                "Micro average": "micro",
                "Weighted average": "weighted",
            }
        )

        metrics["trained_on"] = train_cse

        all_metrics.append(metrics)
    import pdb

    pdb.set_trace()
    return pd.concat(all_metrics)


def old_get_metrics_with_ci(
    mode: str, bootstrap_iter: int = 200, model: Optional[str] = None
):
    """
    Main function
    """

    filename = (
        "note_metrics_ml_all_patients"
        if (mode == "full_on_notes")
        else "entity_metrics_ml_on_gold"
    )

    ALL_CSES = ["Overall"] + CSES

    metrics_with_ci = []
    for train_cse in ALL_CSES:
        df = data.get(filename, cse=train_cse, model=model).reset_index()

        stats = remove_aggregated_label(df.query("label_name != 'Total'"))

        for valid_cse, stat in stats.groupby("cse"):
            metrics = bootstrap(stat, N=bootstrap_iter)
            metrics["trained_on"] = train_cse
            metrics["validated_on"] = valid_cse

            metrics_with_ci.append(metrics)
            import pdb

            pdb.set_trace()
    return pd.concat(metrics_with_ci).reset_index(drop=True)


def get_metrics_with_ci_same_train_val_subcohort(mode: str, bootstrap_iter: int = 200):
    """
    Main function
    """

    filename = (
        "note_metrics_ml_all_patients"
        if (mode == "full_on_notes")
        else "entity_metrics_ml_on_gold"
    )

    ALL_CSES = CSES
    same_train_val = (
        []
    )  # Store stats from data train and validated on the same subcohort
    for train_cse in ALL_CSES:
        df = data.get(filename, cse=train_cse).reset_index()
        stats = remove_aggregated_label(df.query("label_name != 'Total'"))
        same_train_val.append(stats[stats.cse == train_cse])

    stats = pd.concat(same_train_val)
    metrics = bootstrap(stats, N=bootstrap_iter, stratify="cse")

    return metrics.reset_index(drop=True)


def get_collaborative_matrix(
    chosen_metric: str = "F1",
    mode: str = "full_on_notes",
    bootstrap_iter: int = 200,
    compare: str = "micro-macro",
):
    """
    Computes crossed performances
    mode can be either:
    - `full_on_notes`: full NER + ML pipeline at the note level
    - `ml_on_gold_entities`: ML pipeline on gold entities
    """

    assert mode in {"full_on_notes", "ml_on_gold_entities"}
    assert compare in {"micro-macro", "eds-base"}

    HIGHLIGHT_CELLS = [
        [(0, 0), "/\\"],
        [(1, 1), "/"],
        [(2, 2), "/"],
        [(3, 3), "/"],
        [(0, 1), "\\"],
        [(0, 2), "\\"],
        [(0, 3), "\\"],
    ]
    LETTERS = ["A) ", "B) "]

    if compare == "micro-macro":
        methods = ["micro", "macro"]
        titles = [
            f"{letter}{method.capitalize()}-Average {chosen_metric}-Score"
            for letter, method in zip(LETTERS, methods)
        ]
        query = "method=='{method}' & metric == '{chosen_metric}'"
        metrics = get_metrics_with_ci(mode)

    elif compare == "eds-base":
        methods = ["eds", "base"]
        names = ["NLP-ML-CLINICAL", "NLP-ML-PUBLIC"]
        titles = [
            f"{letter}"
            + rf"$\bf{{{name}}}$"
            + f" : Micro-Average {chosen_metric}-Score"
            for letter, name in zip(LETTERS, names)
        ]
        query = "method=='micro' & metric == '{chosen_metric}' & model=='{method}'"
        metrics = []
        for model in methods:
            per_model_metrics = get_metrics_with_ci(mode, model=model)
            per_model_metrics["model"] = model
            metrics.append(per_model_metrics)
        metrics = pd.concat(metrics)

    fig, axes = plt.subplots(1, 2, figsize=[20, 15], sharey=True)

    for method, title, ax in zip(
        methods,
        titles,
        axes,
    ):
        stats = metrics.query(query.format(method=method, chosen_metric=chosen_metric))
        metric = stats.pivot(
            index="validated_on",
            columns="trained_on",
            values="value",
        )

        ci = stats.pivot(
            index="validated_on",
            columns="trained_on",
            values="CI",
        ).to_numpy()

        cax = ax.matshow(metric, cmap="Reds", vmin=84, vmax=97)

        def highlight_cell(x, y, ax=None, **kwargs):
            rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, **kwargs)
            ax = ax or plt.gca()
            ax.add_patch(rect)
            return rect

        for cell, hatch in HIGHLIGHT_CELLS:
            highlight_cell(
                *cell, ax=ax, color="black", linewidth=3, zorder=100, hatch=hatch
            )

        for (i, j), m in np.ndenumerate(metric):
            text = f"{m:0.1f}\n{ci[i,j]}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontdict=dict(fontsize=15),
                bbox=dict(facecolor="white", edgecolor="0.3", alpha=0.8),
                zorder=1000,
            )

        ax.set_xticklabels(
            [""]
            + [Rename.cse[cse_name] + "\n" + "cohort" for cse_name in metric.columns],
            fontdict=dict(fontsize=15),
        )
        ax.set_yticklabels(
            [""]
            + [Rename.cse[cse_name] + "\n" + "cohort" for cse_name in metric.columns],
            fontdict=dict(fontsize=15),
        )
        ax.set_ylabel(
            "Validated on ...",
            loc="center",
            fontdict=dict(weight="bold", fontsize=17, bbox=dict(fill=False)),
        )
        ax.set_xlabel(
            "Trained on ...",
            labelpad=10,
            fontdict=dict(weight="bold", fontsize=17),
            bbox=dict(fill=False),
        )

        ax.yaxis.labelpad = 10
        # ax.xaxis.tick_bottom()
        ax.set_title(
            title,
            fontdict=dict(fontsize=20),  # , weight="bold"),
            pad=30,
        )

    cbar = plt.colorbar(
        cax, ax=axes[:2], orientation="horizontal", fraction=0.046, pad=0.11
    )
    # cbar = plt.colorbar(cax, ax=colorax)#, orientation="horizontal", pad=0.1)
    cbar.set_label(
        "", loc="center", rotation=0, fontdict=dict(fontsize=10, weight="bold")
    )
    cbar.ax.tick_params(labelsize=15)

    return fig
