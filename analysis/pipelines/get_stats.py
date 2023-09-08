import os
from typing import List, Optional

import pandas as pd
import typer
from loguru import logger

from analysis import data
from analysis.collaborative import (
    get_collaborative_matrix,
    get_metrics_with_ci_same_train_val_subcohort,
)
from analysis.format_table import table_1, table_1_a, table_1_b, table_2
from analysis.metrics import (
    compare_all,
    get_entity_metrics,
    get_note_metrics,
    get_side_to_side_metrics,
    prepare_icd10_metrics,
    prepare_note_metrics,
)
from analysis.prevalence import get_prevalence, plot_prevalence
from analysis.stats import get_cohort_stats, get_cohort_stats_per_comorb

MATCH_ON_QUALIFIER = False
BOOTSTRAP_ITER = 200

app = typer.Typer(pretty_exceptions_enable=False)


def step_1(
    stats: pd.DataFrame,
    gold_raw: pd.DataFrame,
    visits: pd.DataFrame,
    texts: pd.DataFrame,
):
    """
    Computes:

    - Cohort statistics (age, N, etc)
    - Note statistics for gold annotations
    - Note statistics for ICD10

    """
    # Stats per cohort
    cohort_stats = get_cohort_stats(
        stats,
    )
    data.save(cohort_stats, "cohort_stats", "Cohort stats")

    cohort_stats_excel = table_1_a(cohort_stats)
    data.save(
        cohort_stats_excel, "cohort_stats_excel", "Cohort stats (EXCEL)", export=True
    )

    # Stats per comorb WITHOUT QUALIFICATION
    cohort_stats_per_comorb = get_cohort_stats_per_comorb(
        gold_raw,
        with_qualification=False,
    )
    data.save(
        cohort_stats_per_comorb,
        "cohort_stats_per_comorb_no_qual",
        "Cohort stats per comorb (NO QUAL)",
    )

    cohort_stats_per_comorb_excel = table_1_b(cohort_stats_per_comorb)
    data.save(
        cohort_stats_per_comorb_excel,
        "cohort_stats_per_comorb_no_qual_excel",
        "Cohort stats (EXCEL NO QUAL)",
        export=True,
    )

    # Concatenated stats
    full_cohort_stats_excel = table_1(
        cohort_stats_excel,
        cohort_stats_per_comorb_excel,
    )
    data.save(
        full_cohort_stats_excel,
        "full_cohort_stats_before_qual_excel",
        "Full cohort stats (EXCEL NO QUAL)",
        export=True,
    )

    # Stats per comorb WITH QUALIFICATION
    cohort_stats_per_comorb = get_cohort_stats_per_comorb(
        gold_raw,
    )
    data.save(
        cohort_stats_per_comorb,
        "cohort_stats_per_comorb",
        "Full cohort stats (EXCEL NO QUAL)",
        export=True,
    )

    cohort_stats_per_comorb_excel = table_1_b(cohort_stats_per_comorb)
    data.save(
        cohort_stats_per_comorb_excel,
        "cohort_stats_per_comorb_excel",
        "Cohort stats (EXCEL)",
        export=True,
    )

    # Concatenated stats
    full_cohort_stats_excel = table_1(
        cohort_stats_excel,
        cohort_stats_per_comorb_excel,
    )
    data.save(
        full_cohort_stats_excel,
        "full_cohort_stats_excel",
        "Full cohort stats (EXCEL)",
        export=True,
    )

    gold_note_stats = prepare_note_metrics(
        df_raw=gold_raw,
        label_type="gold",
    )

    data.save(gold_note_stats, "gold_note_stats", "Gold note stats")

    icd10_note_stats = prepare_icd10_metrics(
        visits,
        texts,
        icd10_prefix="cim10",
    )

    data.save(icd10_note_stats, "icd10_note_stats", "ICD-10 note stats")

    # ICD10 note metrics

    note_metrics = get_note_metrics(
        algo=icd10_note_stats,
        gold=gold_note_stats,
        label_type="icd10",
        group_columns=["label_name", "label_value", "cse"],
    )

    data.save(note_metrics, "note_metrics_icd10", "ICD-10 note-level metrics")

    note_metrics, hyp = get_note_metrics(
        algo=icd10_note_stats,
        gold=gold_note_stats,
        label_type="icd10",
        group_columns=["label_name", "label_value", "cse"],
        with_hyp_testing=True,
        with_bootstrap=BOOTSTRAP_ITER,
    )

    data.save(
        note_metrics, "bs_note_metrics_icd10", "Boostraped ICD-10 note-level metrics"
    )

    data.save(
        hyp,
        "hyp_testing_note_metrics_icd10",
        "All bootstraped sampled / note-level metrics for ICD10",
    )


def step_2(
    algo_raw: pd.DataFrame,
    label_type: str,
    gold_raw: pd.DataFrame,
):
    """
    Computes:

    - Entity metrics (algo VS gold)
    - Note metrics (algo VS gold)
    - Note metrics (algo VS ICD10)

    """

    # Entity metrics
    entity_metrics = get_entity_metrics(
        algo=algo_raw,
        gold=gold_raw,
        label_type=label_type,
        match_on_qualifier=MATCH_ON_QUALIFIER,
    )
    data.save(entity_metrics, f"entity_metrics_{label_type}", "Entity-level metrics")

    entity_metrics_all_patients = get_entity_metrics(
        algo=algo_raw,
        gold=gold_raw,
        label_type=label_type,
        group_columns=["label_name", "label_value", "cse"],
        match_on_qualifier=MATCH_ON_QUALIFIER,
    )
    data.save(
        entity_metrics_all_patients,
        f"entity_metrics_{label_type}_all_patients",
        f"Entity-level metrics for {label_type} (in and outpatients summed)",
    )

    gold_note_stats = data.get("gold_note_stats")
    algo_note_stats = prepare_note_metrics(
        df_raw=algo_raw,
        label_type=label_type,
    )
    data.save(
        algo_note_stats,
        f"algo_note_stats_{label_type}",
        f"Algo note stats ({label_type})",
    )

    note_metrics_all_patients = get_note_metrics(
        algo=algo_note_stats,
        gold=gold_note_stats,
        label_type=label_type,
        group_columns=["label_name", "label_value", "cse"],
    )
    data.save(
        note_metrics_all_patients,
        f"note_metrics_{label_type}_all_patients",
        f"Note-level metrics for {label_type} (in and outpatients summed)",
    )

    bs_note_metrics_all_patients, hyp = get_note_metrics(
        algo=algo_note_stats,
        gold=gold_note_stats,
        label_type=label_type,
        group_columns=["label_name", "label_value", "cse"],
        with_bootstrap=BOOTSTRAP_ITER,
        with_hyp_testing=True,
    )
    data.save(
        bs_note_metrics_all_patients,
        f"bs_note_metrics_{label_type}_all_patients",
        f"Bootstraped note-level metrics for {label_type} (in and outpatients summed)",
    )

    data.save(
        hyp,
        f"hyp_testing_note_metrics_{label_type}",
        f"All bootstraped sampled / note-level metrics for {label_type} (in and outpatients summed)",
    )

    note_metrics = get_note_metrics(
        algo=algo_note_stats,
        gold=gold_note_stats,
        label_type=label_type,
    )
    data.save(
        note_metrics,
        f"note_metrics_{label_type}",
        f"Note-level metrics for {label_type} (in and outpatients separated)",
    )

    bs_note_metrics = get_note_metrics(
        algo=algo_note_stats,
        gold=gold_note_stats,
        label_type=label_type,
        with_bootstrap=BOOTSTRAP_ITER,
    )
    data.save(
        bs_note_metrics,
        f"bs_note_metrics_{label_type}",
        f"Bootstraped note-level metrics for {label_type} (in and outpatients separated)",
    )

    # Excel formatting #

    entity_metrics_excel = table_2(entity_metrics)
    data.save(
        entity_metrics_excel,
        f"entity_metrics_excel_{label_type}",
        "Entity level metrics (EXCEL style)",
        export=True,
    )

    note_metrics_excel = table_2(bs_note_metrics)
    data.save(
        note_metrics_excel,
        f"note_metrics_excel_{label_type}",
        f"Bootstraped note-level metrics for {label_type} (in and outpatients separated, EXCEL style)",
        export=True,
    )

    entity_metrics_excel_all_patients = table_2(entity_metrics_all_patients)
    data.save(
        entity_metrics_excel_all_patients,
        f"entity_metrics_excel_{label_type}_all_patients",
        f"Entity level metrics all patients for {label_type} (EXCEL style)",
        export=True,
    )

    note_metrics_excel_all_patients = table_2(bs_note_metrics_all_patients)
    data.save(
        note_metrics_excel_all_patients,
        f"note_metrics_excel_{label_type}_all_patients",
        f"Bootstraped note-level metrics all patients for {label_type} (EXCEL style)",
        export=True,
    )

    note_metrics_excel_all_patients_per_cse = table_2(
        note_metrics_all_patients, split_per_cse=True
    )
    data.save(
        note_metrics_excel_all_patients_per_cse,
        f"note_metrics_excel_{label_type}_per_cse",
        f"Note-level metrics split per CSE for {label_type} (EXCEL style)",
        export=True,
    )


def step_3(
    ml: pd.DataFrame,
    rule_based: pd.DataFrame,
    no_qual: pd.DataFrame,
    icd10: pd.DataFrame,
):
    """
    Computes:

    - Note metrics (ML VS Rule-based)
    - Note metrics (ML VS Rule-based VS ICD10 VS No qualification)

    """

    all_dataframes = dict(
        ml=ml,
        rule_based=rule_based,
        # no_qual=no_qual, # Uncomment to add no_qual column
        icd10=icd10,
    )

    vs_all = compare_all(all_dataframes)
    data.save(vs_all, "vs_all_excel", "ML vs RULE-BASED vs ICD10 metrics", export=True)

    del all_dataframes["icd10"]

    vs_all_nlp = compare_all(all_dataframes)
    data.save(vs_all_nlp, "vs_all_nlp_excel", "ML vs RULE-BASED metrics", export=True)


def step_4(chosen_metric: str = "F1"):
    """
    Effects of collaborative setting
    """

    for mode in ["full_on_notes"]:  # , "qualifier_on_gold"]:
        matrix = get_collaborative_matrix(
            chosen_metric, mode=mode, bootstrap_iter=BOOTSTRAP_ITER
        )
        data.save(
            matrix,
            f"collaborative_matrix_{mode}",
            f"Collaborative matrix ({mode})",
            export=True,
            bbox_inches="tight",
        )

        same_train_val_metrics = get_metrics_with_ci_same_train_val_subcohort(
            mode=mode,
            bootstrap_iter=BOOTSTRAP_ITER,
        )
        data.save(
            same_train_val_metrics,
            f"same_train_val_metrics_{mode}",
            f"Metrics of model trained and validated on same cohort ({mode})",
            export=True,
        )


def step_5(
    algo: pd.DataFrame,
    gold: pd.DataFrame,
    icd10: pd.DataFrame,
    rule_based: pd.DataFrame,
    texts: pd.DataFrame,
):
    """
    Computes prevalence of eCCI conditions
    """

    side_to_side_metrics = get_side_to_side_metrics(
        algo, gold, icd10, rule_based, group_columns=["label_name", "cse"]
    )
    data.save(
        side_to_side_metrics,
        "side_to_side_metrics",
        "Per note and comorbidity ML vs ICD-10 vs Gold",
    )

    prevalences = get_prevalence(side_to_side_metrics, texts)

    data.save(
        prevalences,
        "prevalences",
        "Prevalence values",
        export=True,
    )

    fig = plot_prevalence(prevalences)

    _ = fig.tight_layout()

    data.save(
        fig,
        "prevalences",
        "Prevalence plot",
        export=True,
        bbox_inches="tight",
    )


@app.command()
def main(
    step: Optional[List[int]] = typer.Option(None, help="Which steps to run"),
):
    cse = os.environ.get("CSE", "Overall")
    model = os.environ.get("MODEL", "eds")

    logger.info(f"Using cse: {cse}")
    logger.info(f"Using BERT model {model}")

    if not step:
        step = [1, 2, 3, 4, 5]

    logger.warning(f"RUNNING FOLLOWING STEPS: {step}")

    stats = data.get("stats", export=True)

    raw = dict(
        ml=data.get("ml"),
        rule_based=data.get("rule_based"),
        gold=data.get("gold"),
        no_qual=data.get("no_qual"),
    )

    note_metrics = dict()  # aggregated

    texts = data.get("notes")
    visits = data.get("visits")

    if 1 in step:
        logger.warning("RUNNING STEP 1")
        step_1(
            stats,
            raw["gold"],
            visits,
            texts,
        )

    note_metrics["icd10"] = data.get("note_metrics_icd10")

    for label_type in ["ml", "rule_based"]:  # , "no_qual"]:
        if 2 in step:
            logger.warning(f"RUNNING STEP 2 ({label_type})")
            step_2(
                raw[label_type],
                label_type,
                raw["gold"],
            )
        note_metrics[label_type] = data.get(f"note_metrics_{label_type}")

    if (4 in step) and (cse == "Overall"):
        logger.warning("RUNNING STEP 4")
        step_4(
            chosen_metric="F1",
        )

    if 5 in step:
        logger.warning("RUNNING STEP 5")

        note_stats = dict()  # per note and per comorb

        note_stats["algo"] = data.get("algo_note_stats_ml")
        note_stats["rule_based"] = data.get("algo_note_stats_rule_based")
        note_stats["gold"] = data.get("gold_note_stats")
        note_stats["icd10"] = data.get("icd10_note_stats")

        step_5(
            algo=note_stats["algo"],
            gold=note_stats["gold"],
            icd10=note_stats["icd10"],
            rule_based=note_stats["rule_based"],
            texts=texts,
        )


if __name__ == "__main__":
    app()
