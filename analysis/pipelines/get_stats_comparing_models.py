import os

import pandas as pd
import typer

from analysis import data
from analysis.format_table import Clean
from analysis.metrics import compare_all, get_qualifier_metrics_on_gold

BOOTSTRAP_ITER = 3

app = typer.Typer(pretty_exceptions_enable=False)


def step_1(
    ml_eds: pd.DataFrame,
    ml_base: pd.DataFrame,
    rule_based: pd.DataFrame,
    icd10: pd.DataFrame,
):
    """
    Computes:

    - Note metrics (ML VS Rule-based)
    - Note metrics (ML VS Rule-based VS ICD10 VS No qualification)

    """

    save_name = "vs_all_eds_base_nlp_excel"

    all_dataframes = dict(
        ml_eds=ml_eds,
        ml_base=ml_base,
        rule_based=rule_based,
    )

    if icd10 is not None:
        all_dataframes["icd10"] = icd10
        save_name = "vs_all_eds_base_excel"

    vs_all = compare_all(all_dataframes)
    vs_all = Clean.run(vs_all)
    data.save(
        vs_all,
        save_name,
        "All models comparison metrics",
        model="compare",
        export=True,
    )


def step_2(eds_qualifier_on_gold: pd.DataFrame, base_qualifier_on_gold: pd.DataFrame):
    """
    Computes qualifier on gold metrics (ML EDS VS ML BASE VS Rule-based)

    """
    qualifier_metrics_on_gold = get_qualifier_metrics_on_gold(
        eds_qualifier_on_gold,
        base_qualifier_on_gold,
    )
    data.save(
        qualifier_metrics_on_gold,
        name="qualifier_metrics_excel_on_gold",
        description="ML metrics on gold entities",
        model="compare",
        export=True,
    )


@app.command()
def main():
    os.environ.get("CSE", "Overall")

    note_metrics = dict(
        ml_eds=data.get("bs_note_metrics_ml", model="eds"),
        ml_base=data.get("bs_note_metrics_ml", model="base"),
        rule_based=data.get("bs_note_metrics_rule_based", model="base"),
        icd10=data.get("bs_note_metrics_icd10", model="base"),
    )

    note_metrics_all_patients = dict(
        ml_eds=data.get("bs_note_metrics_ml_all_patients", model="eds"),
        ml_base=data.get("bs_note_metrics_ml_all_patients", model="base"),
        rule_based=data.get("bs_note_metrics_rule_based_all_patients", model="base"),
    )

    step_1(**note_metrics)
    step_1(icd10=None, **note_metrics_all_patients)

    qualifier_on_gold = dict(
        eds_qualifier_on_gold=data.get("qualifier_on_gold", model="eds"),
        base_qualifier_on_gold=data.get("qualifier_on_gold", model="base"),
    )

    step_2(**qualifier_on_gold)


if __name__ == "__main__":
    app()
