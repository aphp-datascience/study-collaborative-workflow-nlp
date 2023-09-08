import pandas as pd

from analysis.helpers import add_overall


def get_cohort_stats(
    stats: pd.DataFrame,
    with_overall: bool = True,
):
    if with_overall:
        stats = add_overall(stats)

    # Age and Sex

    stats["is_male"] = stats["gender_source_value"] == "M"

    stats = (
        stats.groupby(["cse", "patient_type"])
        .agg(
            age_at_adm=(
                "age_at_adm",
                lambda x: f"{x.mean().round(1)} ({x.std().round(1)})",
            ),
            gender_repartition=(
                "is_male",
                lambda x: f"{int(100*(x.mean()))} - {int(100*(1-x.mean()))}",
            ),
        )
        .reset_index()
    )

    return stats


def get_cohort_stats_per_comorb(
    gold_raw: pd.DataFrame,
    with_overall: bool = True,
    with_qualification: bool = True,
    with_std: bool = False,
):
    gold = gold_raw.copy()
    if with_overall:
        gold = add_overall(gold, on_comorb=True)
    if with_qualification:
        gold = gold[gold.to_keep].copy()

    gold = gold.drop(columns=["to_keep"])
    stats = gold.groupby(["cse", "patient_type", "label_name", "note_id"]).agg(
        N=("upsampled", "size"),
        N_upsampled=(
            "upsampled",
            lambda r: r.astype(bool).sum(),
        ),  # (r is not False).sum()),
        N_normal_sampled=(
            "upsampled",
            lambda r: (~r.astype(bool)).sum(),
        ),  # lambda r: (r is False).sum()),
    )

    stats = stats.groupby(["cse", "patient_type", "label_name"]).agg(
        n_docs_with_entity_normal_sampled=("N_normal_sampled", lambda r: (r > 0).sum()),
        n_docs_with_entity_upsampled=("N_upsampled", lambda r: (r > 0).sum()),
        mean_entity_per_doc=(
            "N_normal_sampled",
            lambda x: (x.sum() / (x > 0).sum()).round(1),
        ),
        std_entity_per_doc=(
            "N",
            lambda x: x.std().round(1),
        ),
    )

    stats = stats.fillna("-")
    stats["n_entity_per_doc"] = stats.apply(
        lambda r: f"{r.mean_entity_per_doc}"
        + (f" ± {r.std_entity_per_doc}" if with_std else ""),
        axis=1,
    )

    stats["n_docs_with_entity"] = stats.apply(
        lambda row: (
            str(row.n_docs_with_entity_normal_sampled)
            if (row.n_docs_with_entity_upsampled == 0)
            else f"{row.n_docs_with_entity_normal_sampled} + {row.n_docs_with_entity_upsampled}"
        ),
        axis=1,
    )

    stats = stats[["n_docs_with_entity", "n_entity_per_doc"]].reset_index()

    # Keeping only number of normally sampled docs for Total
    stats.loc[stats.label_name == "Total", "n_docs_with_entity"] = (
        stats.loc[stats.label_name == "Total", "n_docs_with_entity"]
        .str.split(r" \+ ")
        .str[0]
    )

    return stats
