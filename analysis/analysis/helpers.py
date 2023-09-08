import itertools
import operator
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from tqdm import tqdm

from analysis import DATA_DIR
from analysis.bootstrap import ci_from_list_of_dfs, sample_rows
from analysis.config import RARE_COMORBS
from analysis.format_table import ExcelMapping
from ecci.config import COMORB_CONFIG

palette = sns.color_palette("colorblind")

Colors = {
    "NLP": palette[0],
    "Gold": palette[1],
    "Mentionned": palette[2],
    "Claim": palette[3],
}


class Rename:
    cse = dict(
        cse180032="Cardiology",
        cse200055="Oncology",
        cse200093="Rheumatology",
        Overall="Overall",
    )

    comorbs = {
        "Alcool": "Alcohol Consumption",
        "Cancer": "Solid tumor",
        "Diabète": "Diabetes",
        "Démence": "Dementia",
        "Hémiplégie": "Hemiplegia",
        "IDM": "Myocardial infarction",
        "Insuff. Card. Cong.": "Congestive heart failure",
        "Leucemie": "Leukemia",
        "Lymphome": "Lymphoma",
        "Maladies Cérébro-Vasc.": "Cerebrovascular disease",
        "Maladies Pulmo. Chron.": "Chronic pulmonary disease",
        "Maladies Vasc. Periph.": "Peripheral vascular disease",
        "Maladies des Tissus Conj.": "Rheumatologic disease",
        "Maladies du Foie": "Liver disease",
        "Maladies du Rein": "Renal disease",
        "SIDA": "AIDS",
        "Tabagisme": "Tobacco consumption",
        "Ulcères gastriques": "Peptic ulcer disease",
    }

    include = [cse, comorbs]
    mapping = {
        k: v
        for k, v in itertools.chain.from_iterable(
            map(operator.methodcaller("items"), include)
        )
    }

    @classmethod
    def run(cls, s):
        return cls.mapping.get(s, s)


def stats_sampling_for_bootstrap(
    stats: pd.DataFrame,
    with_total,
    with_specificity,
):
    """
    Declared here for multiprocessing
    """
    copied_stats = stats[["TP", "FN", "FP"]].apply(sample_rows, axis=1)
    return add_classification_metrics(
        copied_stats,
        with_total=with_total,
        with_bootstrap=False,
        with_specificity=with_specificity,
    )


def add_notes_infos(
    data: pd.DataFrame,
    notes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adds a snippet and lexical variant, and removes the `note_text` column

    Parameters
    ----------
    data : pd.DataFrame
        Output of the NLP pipeline
    notes : pd.DataFrame
        With `note_id` and `note_text` columns

    Returns
    -------
    pd.DataFrame
    """
    keep_cols = list(
        set(
            [
                "person_id",
                "note_id",
                "cse",
                "patient_type",
                "note_text",
                "visit_occurrence_id",
            ]
        )
        & set(notes.columns)
    )
    merge_cols = list(set(data.columns) & set(notes.columns))

    df = data.merge(
        notes[keep_cols],
        on=merge_cols,
        how="inner",
    )

    df["snippet"] = df.apply(
        lambda r: r.note_text[
            max(0, r.offset_begin - 200) : min(len(r.note_text), r.offset_end + 200)
        ],
        axis=1,
    )

    df["lexical_variant"] = df.apply(
        lambda r: r.note_text[r.offset_begin : r.offset_end],
        axis=1,
    )

    # Removing now-useless text
    del df["note_text"]

    return df


def add_overall(
    data: pd.DataFrame,
    on_cse: bool = True,
    on_comorb: bool = False,
) -> pd.DataFrame:
    """
    Duplicates the `data` DataFrame with:
    - an "Overall" cse tag AND/OR
    - a "Total" label_name

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame, with N rows

    Returns
    -------
    pd.DataFrame
        The output DataFrame with added rows
    """
    if on_cse:
        if "cse" in data.columns:
            data_overall = data.copy()
            data["cse"] = "Overall"
            data = pd.concat([data, data_overall])

    if on_comorb:
        if "label_name" in data.columns:
            # data_total = remove_aggregated_label(data.copy())
            data_total = data.copy()
            data_total = data_total[~data_total.label_name.isin(RARE_COMORBS)]
            data_total["label_name"] = "Total"
            data_total["label_value"] = "1"
            data = pd.concat([data, data_total])

    return data


def add_aggregated_label(
    data: pd.DataFrame,
):
    """
    Add a new label_value (3) for non-binary comorbidities to aggregate
    """

    multiple_status = [
        c["label_name"] for c in COMORB_CONFIG if c["selection_type"] == "list"
    ]

    multiple = data[data.label_name.isin(multiple_status)].copy()
    multiple["label_value"] = "3"

    return pd.concat([data, multiple])


def remove_aggregated_label(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    For comorbidities with multiple status, only keep row with label_value == 3
    """

    if ("label_value" not in data.columns) or (data.label_value.nunique() == 1):
        return data

    multiple_status = [
        c["label_name"] for c in COMORB_CONFIG if c["selection_type"] == "list"
    ]  # + ["Total"]

    single = data[~data.label_name.isin(multiple_status)]
    multiple = data[data.label_name.isin(multiple_status)].copy()
    multiple = multiple[multiple.label_value.str.contains(r"\b(3|average)")]
    multiple["label_value"] = "1"

    return pd.concat([single, multiple])


def add_classification_metrics(
    stats: pd.DataFrame,
    with_total=["cse", "patient_type"],
    with_bootstrap=False,
    with_specificity=False,
    with_hyp_testing=False,
) -> pd.DataFrame:
    """
    Adds "F1", "PPV", "Support" and "Sensitivity" columns.
    If any of those columns already exists, keeps the existing one

    Parameters
    ----------
    stats : pd.DataFrame
        With columns "TP", "FP" and "FN".
    with_bootstrap
        If set to an integer, runs a bootstraping with that many steps
    with_spceficity
        If set to an integer, add a "Specificity" columns and uses the
        provided integer as the number of total cases (= TP+FP+FN+TN)
    Returns
    -------
    pd.DataFrame
    """
    if "Support" not in stats.columns:
        stats["Support"] = stats["TP"] + stats["FN"]
    if "PPV" not in stats.columns:
        stats["PPV"] = 100 * stats["TP"] / (stats["TP"] + stats["FP"])
    if "Sensitivity" not in stats.columns:
        stats["Sensitivity"] = 100 * stats["TP"] / stats["Support"]
    if "F1" not in stats.columns:
        stats["F1"] = (
            2
            * (stats["PPV"] * stats["Sensitivity"])
            / (stats["PPV"] + stats["Sensitivity"])
        )

    if with_specificity and ("Specificity" not in stats.columns):
        if "TN" not in stats.columns:
            stats["TN"] = 999
            for index, row in stats.iterrows():
                if ("inpatient" in row.name) or ("outpatient" in row.name):
                    N = 50
                else:
                    N = 100
                if "Overall" in row.name:
                    N *= 3
                stats.loc[index, "TN"] = N - (row["TP"] + row["FP"] + row["FN"])

        stats["Specificity"] = 100 * stats["TN"] / (stats["TN"] + stats["FP"])
        assert stats.query("TN == 999").empty

    if with_total:
        stats = add_avg_metrics(
            stats, group_cols=with_total, with_specificity=with_specificity
        )

    if with_bootstrap:
        # all_stats = []
        # for _ in tqdm(range(with_bootstrap)):
        #     copied_stats = stats[["TP", "FN", "FP"]].apply(sample_rows, axis=1)
        #     all_stats.append(
        #         add_classification_metrics(
        #             copied_stats,
        #             with_total=with_total,
        #             with_bootstrap=False,
        #             with_specificity=with_specificity,
        #         )
        #     )
        all_stats = Parallel(
            n_jobs=-1,
            backend="multiprocessing",
        )(
            delayed(stats_sampling_for_bootstrap)(
                stats,
                with_total,
                with_specificity,
            )
            for i in tqdm(range(with_bootstrap), total=with_bootstrap)
        )
        ci = ci_from_list_of_dfs(all_stats)

        ci = ci.applymap(
            lambda list_metrics: f"({np.quantile(list_metrics, q=0.025):.1f} - {np.quantile(list_metrics, q=0.975):.1f})"
        )

        stats_formatted = stats.round(1).astype(str) + " " + ci

        if with_hyp_testing:
            return (stats_formatted, all_stats)

        return stats_formatted

    return stats


def add_avg_metrics(
    df: pd.DataFrame,
    group_cols=["cse", "patient_type"],
    macro_only: bool = False,
    with_specificity: bool = False,
):
    METRICS = ["PPV", "Sensitivity", "F1"]
    COUNTS = ["TP", "FP", "FN"]
    if with_specificity:
        METRICS += ["Specificity"]
        COUNTS += ["TN"]

    index = list(df.index.names)

    df = df.copy().reset_index()
    df = df[~df.label_name.isin(["Total"])]
    data = df.copy()
    data = data[~data.label_name.isin(RARE_COMORBS)]
    df = df.set_index(index)

    group_cols = [c for c in group_cols if c in df.reset_index().columns]
    if not group_cols:
        group_cols = ["dummy"]
        data["dummy"] = "dummy"

    micro = pd.DataFrame()

    if not macro_only:
        micro = (
            remove_aggregated_label(data.copy())
            .groupby(group_cols)
            .sum()
            .reset_index()[group_cols + COUNTS]
        )

        micro = add_classification_metrics(
            micro.set_index(group_cols),
            with_total=False,
            with_specificity=with_specificity,
        )
        micro = pd.concat({"Micro average": micro}, names=["label_value"])

    macro = remove_aggregated_label(data.copy())

    weighted = macro.copy()

    macro = macro.groupby(group_cols).mean()
    macro = pd.concat({"Macro average": macro}, names=["label_value"])

    for metric in METRICS:
        weighted[metric] = weighted[metric] * weighted["Support"]

    weighted = weighted.groupby(group_cols).sum()
    for metric in METRICS:
        weighted[metric] = weighted[metric] / weighted["Support"]

    weighted = pd.concat({"Weighted average": weighted}, names=["label_value"])

    if len(index) == 1:  # Single index columns
        macro = macro.reset_index(drop=True)
        macro["label_name"] = "Macro average"
        macro.set_index("label_name", inplace=True)

        micro = micro.reset_index(drop=True)
        micro["label_name"] = "Micro average"
        micro.set_index("label_name", inplace=True)

        weighted = weighted.reset_index(drop=True)
        weighted["label_name"] = "Weighted average"
        weighted.set_index("label_name", inplace=True)

    total_avg = pd.concat([micro, macro, weighted])

    if len(index) != 1:
        total_avg = pd.concat({"Total": total_avg}, names=["label_name"])

    result = pd.concat(
        [
            df,
            total_avg,
        ]
    )

    return result


def get_paper_data():
    sources = ["eds", "compare"]

    with pd.ExcelWriter(DATA_DIR / "paper_data" / "tables.xlsx") as writer:
        for source in sources:
            excels = (DATA_DIR / source / "Overall" / "export").glob("*excel*")

            for excel in excels:
                print(excel)
                excel = Path(excel)
                pd.read_pickle(excel).to_excel(
                    writer, sheet_name=ExcelMapping.get(excel.stem)
                )

            figures = (DATA_DIR / source / "Overall" / "export").glob("*png*")

            for fig in figures:
                print(fig)
                fig = Path(fig)
                shutil.copyfile(
                    fig,
                    DATA_DIR / "paper_data" / fig.name,
                )
