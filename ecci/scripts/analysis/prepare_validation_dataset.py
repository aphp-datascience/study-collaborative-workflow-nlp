import re

import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T
from confection import Config
from edstoolbox import SparkApp
from loguru import logger

from ecci import BASE_DIR, DATA_DIR
from ecci.config import DB
from ecci.io import SparkData
from ecci.stats.validation_dataset import prepare_for_annotation

app = SparkApp("ecci-prepare-annotation")


@app.submit
def main(spark, sql, config: Config):
    #     config = Config().from_disk(
    #         BASE_DIR.parent / "scripts" / "analysis" / "config.cfg"
    #     )["validation_dataset"]
    #     if not config["run"]:
    #         logger.info("Skipping validation_dataset")
    #         return

    #     texts, entities = [], []
    #     for patient_type in config["patient_type"]:
    #         logger.info(f"Getting dataset for {patient_type}")

    #         note_nlp = pd.read_pickle(
    #             DATA_DIR / f"validation_dataset_{patient_type}.pickle"
    #         )
    #         notes = pd.read_pickle(
    #             DATA_DIR / f"validation_dataset_notes_{patient_type}.pickle"
    #         )

    #         prepared_note_nlp, prepared_notes = prepare_for_annotation(
    #             note_nlp=note_nlp,
    #             notes=notes,
    #             kept_comorbs=config["kept_comorbs"],
    #             snippet_length=config["snippet_length"],
    #             max_notes=config["max_notes"],
    #             patient_type=patient_type,
    #             seed=config["seed"],
    #         )

    #         entities.append(prepared_note_nlp)
    #         texts.append(prepared_notes)

    #         prepared_note_nlp.to_pickle(
    #             DATA_DIR / f"prepared_validation_dataset_{patient_type}.pickle"
    #         )
    #         prepared_notes.to_pickle(
    #             DATA_DIR / f"prepared_validation_dataset_notes_{patient_type}.pickle"
    #         )

    #     entities = pd.concat(entities)
    #     texts = pd.concat(texts)

    #     entities.to_pickle(DATA_DIR / "rare_comorbs/entities.pickle")

    #     texts.to_pickle(DATA_DIR / "rare_comorbs/texts.pickle")

    # Adding informations

    notes = pd.read_pickle(DATA_DIR / "rare_comorbs/texts.pickle")

    notes.to_pickle(DATA_DIR / "rare_comorbs/texts_bkp.pickle")

    notes["patient_type"] = notes.note_id.str.split("-").str[1]

    data = list(notes[["old_note_id", "patient_type"]].itertuples(index=False))

    schema = T.StructType(
        [
            T.StructField("note_id", T.StringType(), True),
            T.StructField("patient_type", T.StringType(), True),
        ]
    )
    df = spark.createDataFrame(data=data, schema=schema).cache()

    print(df.count())

    S = SparkData(
        spark=spark,
        patient_type="inpatient",  # temporary
        dataset_type="test",
    )

    notes_spark = S.orbis_note.join(
        F.broadcast(df),
        on="note_id",
        how="inner",
    ).join(
        S.visit_occurrence,
        on=["person_id", "visit_occurrence_id"],
        how="left",
    )

    print(notes_spark.cache().count())

    notes_enriched = notes_spark.toPandas()

    notes.merge(
        notes_enriched[
            [
                "visit_start_datetime",
                "visit_end_datetime",
                "visit_source_value",
                "note_id",
                "person_id",
                "visit_occurrence_id",
                "note_datetime",
                "note_class_source_value",
            ]
        ].rename(columns=dict(note_id="old_note_id")),
        on="old_note_id",
    ).to_pickle(DATA_DIR / "rare_comorbs/texts.pickle")

    S.condition_occurrence = (
        S.condition_occurrence.join(
            F.broadcast(notes_spark.select(["person_id", "visit_occurrence_id"])),
            on=["person_id", "visit_occurrence_id"],
        )
    ).cache()

    S.get_notes()

    comorbs_cols = [c for c in S.visits_pd.columns if "cim10" in c]

    missing = notes_enriched[
        ["visit_occurrence_id", "person_id", "visit_start_datetime"]
    ]
    missing = missing[
        ~missing.visit_occurrence_id.isin(S.visits_pd.visit_occurrence_id)
    ]

    pat = re.compile(r"cim10_(?!(AIDS|hemi))")

    other_comorbs_cols = [c for c in S.visits_pd.columns if pat.search(c)]

    for comorb in other_comorbs_cols:
        S.visits_pd[comorb] = "0:Absence"

    all_visits = pd.concat([S.visits_pd, missing])

    for comorb in comorbs_cols:
        all_visits[comorb] = all_visits[comorb].fillna("0:Absence")

    all_visits.to_pickle(DATA_DIR / "rare_comorbs" / "visits.pickle")


if __name__ == "__main__":
    app.run()
