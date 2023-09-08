import os

import pandas as pd
import pyspark.sql.functions as F

from ecci import DATA_DIR
from ecci.io import SparkData


def get_cohort_stats(spark):
    all_visits = []
    cse = os.environ["USER"]

    for patient_type in ["inpatient", "outpatient"]:
        cohort_notes = pd.read_pickle(DATA_DIR / patient_type / "texts.pickle")
        cohort_notes["patient_type"] = patient_type

        cohort_notes["cse"] = cse

        data_getter = SparkData(
            spark,
            patient_type=patient_type,
        )

        visits = data_getter.visit_occurrence
        person = data_getter.person

        visits = (
            visits.filter(
                F.col("visit_occurrence_id").isin(
                    set(cohort_notes.visit_occurrence_id.unique())
                )
            )
            .join(
                person,
                on="person_id",
                how="inner",
            )
            .toPandas()
        )

        visits = visits.merge(
            cohort_notes[
                ["visit_occurrence_id", "cse", "patient_type"]
            ].drop_duplicates(),
            on="visit_occurrence_id",
            how="inner",
        )

        visits["age_at_adm"] = (
            visits["visit_start_datetime"] - visits["birth_datetime"]
        ).astype("<m8[Y]")

        all_visits.append(visits)

    all_visits = pd.concat(all_visits)

    return all_visits[["cse", "patient_type", "age_at_adm", "gender_source_value"]]
