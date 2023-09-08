"""
Should run add_aggregated BEFORE add_overall
"""

from typing import Dict, List, Union

import pandas as pd

from analysis.format_table import Clean
from analysis.helpers import (
    add_aggregated_label,
    add_avg_metrics,
    add_classification_metrics,
    add_overall,
    remove_aggregated_label,
)
from ecci.config import COMORB_CONFIG


def prepare(
    df: pd.DataFrame,
    label: Union[str, List[str]] = ["label_name", "label_value"],
    label_type: str = "gold",
    with_overall: bool = True,
    match_on_qualifier: bool = True,
):
    """
    if match_on_qualifier is True, we don't discard entities with `to_keep = False` and
    consider and entity with correct label and qualifier (even if to_keep=False) as TP
    """

    assert label_type in {"gold", "ml", "rule_based", "no_qual"}

    if not match_on_qualifier:
        df = df[df.to_keep].copy()
    else:
        df = df.copy()
        label = list(set(label) | set(["to_keep"]))

    # Add overall:
    if with_overall:
        data = add_overall(df)

    # Add unique id
    data["idx"] = list(range(len(data)))

    # Create interval
    data["span"] = data.apply(
        lambda r: pd.Interval(left=r.offset_begin, right=r.offset_end, closed="both"),
        axis=1,
    )

    label = [label] if isinstance(label, str) else label

    cols = {c: f"{label_type}_{c}" for c in ["idx", "span"]}

    data.rename(columns=cols, inplace=True)

    data = data[label + ["note_id", "cse", "patient_type"] + list(cols.values())]

    return data


def get_qualifier_metrics_on_gold(
    eds_qualifier_on_gold: pd.DataFrame, base_qualifier_on_gold: pd.DataFrame
):
    qualifier_on_gold = eds_qualifier_on_gold.copy()

    base_to_keep = base_qualifier_on_gold["to_keep"]

    rb_to_keep = ~(
        base_qualifier_on_gold.negation
        | base_qualifier_on_gold.hypothesis
        | base_qualifier_on_gold.family
    ).copy()

    qualifier_on_gold["ml_base_to_keep"] = base_to_keep
    qualifier_on_gold["rule_based_to_keep"] = rb_to_keep

    qualifier_on_gold = qualifier_on_gold.rename(columns=dict(to_keep="ml_eds_to_keep"))

    qualifier_on_gold[
        ["ml_eds_to_keep", "ml_base_to_keep", "rule_based_to_keep", "gold_to_keep"]
    ] = qualifier_on_gold[
        ["ml_eds_to_keep", "ml_base_to_keep", "rule_based_to_keep", "gold_to_keep"]
    ].astype(
        bool
    )

    df = add_overall(add_aggregated_label(qualifier_on_gold)).query("cse=='Overall'")

    dfs = []

    for method in ["ml_eds", "ml_base", "rule_based"]:
        df["TP"] = df[f"{method}_to_keep"] & df.gold_to_keep
        df["FP"] = df[f"{method}_to_keep"] & (~df.gold_to_keep)
        df["FN"] = (~df[f"{method}_to_keep"]) & df.gold_to_keep

        # Possibility to add "patient_type" in the grouping variable + in the "with_total" list here
        df_grouped = df.groupby(["label_name", "label_value", "cse"])[
            ["TP", "FN", "FP"]
        ].sum()

        df_grouped = add_classification_metrics(df_grouped, with_total=["cse"])
        df_grouped["method"] = method

        dfs.append(df_grouped)

    df = pd.concat(dfs).reset_index()

    return Clean.run(
        df.pivot(
            index=["label_name", "label_value"],
            columns=["method"],
            values=["F1", "PPV", "Sensitivity"],
        )
    ).applymap(lambda v: round(v, 1))


def get_entity_metrics(
    algo: pd.DataFrame,
    gold: pd.DataFrame,
    label_type: str = "ml",
    group_columns: List[str] = [
        "label_name",
        "label_value",
        "cse",
        "patient_type",
    ],
    match_on_qualifier: bool = True,
):
    """
    if match_on_qualifier is True, we don't discard entities with `to_keep = False` and
    consider and entity with correct label and qualifier (even if to_keep=False) as TP
    """
    algo = add_aggregated_label(algo)
    gold = add_aggregated_label(gold)

    algo = prepare(algo, label_type=label_type, match_on_qualifier=match_on_qualifier)
    gold = prepare(gold, label_type="gold", match_on_qualifier=match_on_qualifier)

    # cartesian product
    product = gold.merge(
        algo,
        on=(
            group_columns
            + [
                "note_id",
            ]
        ),
        how="outer",
        suffixes=("_gold", f"_{label_type}"),
    )

    # Dummy interval which will never overlap with a real one
    product[f"{label_type}_span"].fillna(
        pd.Interval(-1, -1, closed="both"), inplace=True
    )
    product["gold_span"].fillna(pd.Interval(-1, -1, closed="both"), inplace=True)

    def check_tp(row):
        mask = row[f"{label_type}_span"].overlaps(row["gold_span"])
        if match_on_qualifier:
            mask = mask & (row[f"to_keep_{label_type}"] == row["to_keep_gold"])
        return mask

    # TP: overlap with a gold entity
    product["tp"] = product.apply(check_tp, axis=1)
    # Precision
    precision = (
        product.groupby(group_columns + [f"{label_type}_idx"])["tp"].any().reset_index()
    )
    precision = precision.groupby(group_columns).agg(
        TP=("tp", "sum"),
        TP_plus_FP=("tp", "size"),
    )

    precision["PPV"] = 100 * (precision.TP / precision.TP_plus_FP)
    precision["FP"] = precision["TP_plus_FP"] - precision["TP"]

    recall = product.groupby(group_columns + ["gold_idx"])["tp"].any().reset_index()
    recall = recall.groupby(group_columns).agg(
        tp=("tp", "sum"),
        Support=("tp", "size"),
    )

    recall["Sensitivity"] = 100 * (recall.tp / recall.Support)

    stats = precision[["PPV", "TP", "FP"]].merge(
        recall[["Sensitivity", "Support"]],
        left_index=True,
        right_index=True,
        how="outer",
    )
    stats["Support"] = stats["Support"].fillna(0)

    stats["FN"] = stats["Support"] - stats["TP"]

    stats["F1"] = (
        2
        * (stats["PPV"] * stats["Sensitivity"])
        / (stats["PPV"] + stats["Sensitivity"])
    )

    stats = add_avg_metrics(
        stats,
        macro_only=False,
    )

    return stats


def get_note_metrics(
    algo: pd.DataFrame,
    gold: pd.DataFrame,
    label_type: str = "ml",
    group_columns: List[str] = [
        "label_name",
        "label_value",
        "cse",
        "patient_type",
    ],
    with_bootstrap=False,
    with_hyp_testing=False,
):
    """
    (algo, gold) : output of prepare_note_level_metrics
    """

    if label_type == "icd10":
        gold = gold.query("patient_type == 'inpatient'").copy()

    stats = pd.concat([algo, gold])

    stats = (
        stats.groupby(group_columns + ["note_id"])
        .agg(
            TP=("source", lambda x: (label_type in x.values) and ("gold" in x.values)),
            FP=(
                "source",
                lambda x: (label_type in x.values) and ("gold" not in x.values),
            ),
            FN=(
                "source",
                lambda x: (label_type not in x.values) and ("gold" in x.values),
            ),
        )
        .reset_index()
    )

    stats = stats.groupby(group_columns)[["TP", "FP", "FN"]].sum()

    stats = add_classification_metrics(
        stats,
        with_total=["cse", "patient_type"],
        with_bootstrap=with_bootstrap,
        with_specificity=True,
        with_hyp_testing=with_hyp_testing,
    )

    if with_hyp_testing:
        # we have a tuple
        stats, hyp = stats

        return (
            stats,
            [
                h[
                    [
                        "Support",
                        "PPV",
                        "Sensitivity",
                        "Specificity",
                        "F1",
                        "TP",
                        "FP",
                        "FN",
                        "TN",
                    ]
                ]
                for h in hyp
            ],
        )

    return stats[
        ["Support", "PPV", "Sensitivity", "Specificity", "F1", "TP", "FP", "FN", "TN"]
    ]


def prepare_note_metrics(
    df_raw: pd.DataFrame,
    label_type: str = "ml",
    group_columns: List[str] = ["label_name", "cse", "patient_type"],
    with_aggregated: bool = True,
):
    df = df_raw[df_raw.to_keep].copy()

    df = (
        df.groupby(group_columns + ["note_id"])
        .label_value.max()
        .to_frame()
        .reset_index()
    )
    if with_aggregated:
        df = add_aggregated_label(df)
    df = add_overall(df)

    df["source"] = label_type

    return df


def prepare_icd10_metrics(
    visits,
    texts,
    icd10_prefix="cim10",
    with_overall: bool = True,
    with_aggregated: bool = True,
):
    assert icd10_prefix in {"cim10", "all_cim10"}

    df = visits.merge(
        texts[["note_id", "visit_occurrence_id"]],
        on="visit_occurrence_id",
        how="inner",
    )[
        ["note_id", "cse"] + [c for c in visits.columns if c.startswith(icd10_prefix)]
    ].drop_duplicates()

    renaming = {comorb["pipe_name"]: comorb["label_name"] for comorb in COMORB_CONFIG}

    stats = pd.wide_to_long(
        df,
        icd10_prefix,
        ["note_id", "cse"],
        "comorb",
        sep="_",
        suffix=".*",
    ).reset_index()

    stats.columns = ["note_id", "cse", "label_name", "label_value"]
    stats["label_value"] = stats["label_value"].str[:1]

    stats = stats[stats.label_value != "0"]

    stats["label_name"] = stats["label_name"].replace(renaming)

    if with_aggregated:
        stats = add_aggregated_label(stats)
    stats["source"] = "icd10"

    if with_overall:
        stats = add_overall(stats)

    return stats


def get_side_to_side_metrics(
    algo: pd.DataFrame,
    gold: pd.DataFrame,
    icd10: pd.DataFrame,
    rule_based: pd.DataFrame,
    label_type: str = "ml",
    group_columns: List[str] = ["label_name", "label_value", "cse"],
):
    stats = pd.concat(
        [
            algo.query("patient_type == 'inpatient'").drop(columns="patient_type"),
            gold.query("patient_type == 'inpatient'").drop(columns="patient_type"),
            rule_based.query("patient_type == 'inpatient'").drop(
                columns="patient_type"
            ),
            icd10,
        ]
    )

    stats = (
        stats.groupby(group_columns + ["note_id"])
        .agg(
            gold=("source", lambda x: ("gold" in x.values)),
            nlp=("source", lambda x: (label_type in x.values)),
            rule_based=("source", lambda x: ("rule_based" in x.values)),
            claim=("source", lambda x: ("icd10" in x.values)),
        )
        .reset_index()
    )

    return stats


def get_icd10_vs_algo_metrics(
    side_to_side_metrics: pd.DataFrame,
    group_columns: List[str] = ["label_name", "label_value", "cse"],
):
    """
    side_to_side_metrics: output of `get_side_to_side_metrics` function
    """

    stats = side_to_side_metrics[side_to_side_metrics.gold].copy()

    stats = stats.groupby(group_columns)[
        ["nlp_not_claim", "claim_not_nlp", "gold"]
    ].sum()

    stats["% NLP Only"] = (100 * (stats["nlp_not_claim"] / stats["gold"])).map(
        "{:,.2f}".format
    ) + " %"
    stats["% Claim Only"] = (100 * (stats["claim_not_nlp"] / stats["gold"])).map(
        "{:,.2f}".format
    ) + " %"

    return stats[["% NLP Only", "% Claim Only", "gold"]].rename(
        columns={"gold": "Support"}
    )


def compare_all(
    data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Compares global metrics between inputs

    Returns
    -------
    pd.DataFrame
    """

    results = []

    query = "cse=='Overall' "
    if "icd10" in data:
        data["icd10"]["patient_type"] = "inpatient"
        query += "& patient_type=='inpatient'"

    for k, df in data.items():
        df = remove_aggregated_label(df.copy().reset_index()).query(query)

        df.loc[df.label_name == "Total", "label_name"] = df.loc[
            df.label_name == "Total", "label_value"
        ]
        df_grouped = df

        df_grouped = pd.melt(
            df_grouped,
            id_vars=["label_name"],  # , "Support"],
            value_vars=["PPV", "Sensitivity", "Specificity", "F1"],
            var_name="Metric",
            value_name="Value",
        )
        df_grouped["label_type"] = k

        results.append(df_grouped)

    df = pd.concat(results)

    df = df.pivot(
        index=["label_name"],  # , "Support"],
        columns=["Metric", "label_type"],
        values="Value",
    )

    # Keeping CI for average only

    df.loc[df.index.str.contains("average"), :] = df.loc[
        df.index.str.contains("average"), :
    ].applymap(lambda s: "\n".join(s.split(" ", maxsplit=1)))

    df.loc[~df.index.str.contains("average"), :] = df.loc[
        ~df.index.str.contains("average"), :
    ].applymap(lambda s: s.split(" ", maxsplit=1)[0])

    return Clean.run(df)
