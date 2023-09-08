import itertools
import operator

import pandas as pd

from ecci.config import COMORB_CONFIG


class ExcelMapping:
    mapping = dict(
        full_cohort_stats_excel="MAIN - Table 1 - Cohort stats",
        note_metrics_excel_ml_all_patients="MAIN - Table 2 - Note metrics (NLP-ML-CLINICAL)",
        vs_all_eds_base_excel="MAIN - Table 3 - Inpatient comparison",
        full_cohort_stats_before_qual_excel="SUPPL - Table 4 - Cohort stats before Qual",
        entity_metrics_excel_ml_all_patients="SUPPL - Table 5 - Entity metrics",
        qualifier_metrics_excel_on_gold="SUPPL - Table 6 - Qualifier metrics",
        note_metrics_excel_ml="SUPPL - Table 7 - Note metrics per stay type",
        note_metrics_excel_ml_per_cse="SUPPL - Table 8 - Note metrics per CSE",
    )

    @classmethod
    def get(cls, name):
        return cls.mapping.get(name, f"X - {name}")


class Clean:

    """
    Class to prepare export: sort columns and rows
    """

    cse = {
        "Overall": 99,
        "cse180032": 0,
        "cse200055": 2,
        "cse200093": 3,
    }
    misc = {"Total": 99}

    source = {
        "Gold": 0,
        "NLP": 2,
        "Claim": 3,
    }

    label_type = {
        "ml_eds": 0,
        "ml_base": 1,
        "ml": 2,
        "rule_based": 3,
        "no_qual": 4,
    }

    label_name = {c["label_name"]: c["position"] for c in COMORB_CONFIG}
    label_name["Total"] = 90

    label_name["Micro average"] = 90
    label_name["Macro average"] = 91
    label_name["Weighted average"] = 92

    patient_type = {"inpatient": 0, "outpatient": 1}

    include = [cse, label_name, patient_type, misc, source, label_type]
    mapping = {
        k: v
        for k, v in itertools.chain.from_iterable(
            map(operator.methodcaller("items"), include)
        )
    }

    @classmethod
    def key(cls, s):
        tmp = pd.Series(s)
        return tmp.replace(cls.mapping)

    @classmethod
    def sort_columns(cls, df):
        df = df.sort_index(axis=1, key=cls.key)
        return df

    @classmethod
    def sort_index(cls, df):
        df = df.sort_index(axis=0, key=cls.key)
        return df

    @classmethod
    def remove_irrelevant_level(cls, df):
        if len(set([c[0] for c in df.columns])) == 1:
            df.columns = df.columns.droplevel()
        return df

    @classmethod
    def run(cls, df):
        tmp = df.copy()
        tmp = cls.remove_irrelevant_level(tmp)
        tmp = cls.sort_columns(tmp)
        tmp = cls.sort_index(tmp)

        return tmp


def table_1_a(df):
    """
    Age and Sex repartition
    """

    # To long format

    long_format = pd.melt(
        df,
        id_vars=["cse", "patient_type"],
        var_name="Metric",
        value_name="Value",
    )
    long_format["metric_type"] = "Descriptive data"

    # To correct format

    final_stats = long_format.pivot(
        index=["metric_type", "Metric"],
        columns=["cse", "patient_type"],
        values=["Value"],
    )

    # Adding total number of records

    total_number_records = final_stats.iloc[-1].copy()
    total_number_records.name = ("Descriptive data", "total_number_records")
    total_number_records.loc[:] = 50

    final_stats = final_stats.append(total_number_records)

    return Clean.run(final_stats)


def table_1_b(df):
    """
    Per comorb statistics
    """

    # Aggregate to single column
    df["stat"] = df.apply(
        lambda r: f"{r.n_docs_with_entity} ({r.n_entity_per_doc})", axis=1
    )

    # To correct format
    final_stats = df.pivot(
        index=["label_name"],
        columns=["cse", "patient_type"],
        values=["stat"],
    ).fillna("0")

    return Clean.run(final_stats)


def table_1(df_a, df_b):
    df_b = pd.concat({"eCCI condition": df_b}, names=["metric_type"])
    final_stats = pd.concat([df_a, df_b])
    final_stats.index.names = ["", ""]

    return Clean.run(final_stats).droplevel(0)


def table_2(df, split_per_cse: bool = False):
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    patient_type = ["patient_type"] if "patient_type" in df.columns else []

    if split_per_cse:
        df = df.query("cse != 'Overall'")
    else:
        df = df.query("cse == 'Overall'")

    # To long format

    value_vars = set(["PPV", "Sensitivity", "F1", "Specificity"]) & set(df.columns)

    long_format = pd.melt(
        df,
        id_vars=["label_name", "label_value", "cse"] + patient_type,
        value_vars=value_vars,  # Could add "Support" here
        var_name="Metric",
        value_name="Value",
    )

    # To correct format

    final_stats = (
        long_format.pivot(
            index=["label_name", "label_value"],
            columns=["cse", "Metric"] + patient_type,
            values=["Value"],
        )
        .round(1)
        .fillna("-")
    )

    return Clean.run(final_stats)


def table_3(df):
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    # To long format

    long_format = pd.melt(
        df,
        id_vars=["label_name", "label_value", "cse"],
        value_vars=["% NLP Only", "% Claim Only", "Support"],
        var_name="Metric",
        value_name="Value",
    )

    # To correct format

    final_stats = (
        long_format.pivot(
            index=["label_name", "label_value"],
            columns=["cse", "Metric"],
            values=["Value"],
        )
        .round(1)
        .fillna("-")
    )

    return Clean.run(final_stats)
