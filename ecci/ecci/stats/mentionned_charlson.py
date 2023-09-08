import pyspark.sql.functions as F
from edsnlp.processing import pipe

from ecci.io.connectors import SparkData
from ecci.ner import get_nlp


def get_mentionned_charlson(spark):
    """
    Extract all notes with a charslon score mention.
    For those visits, adds the claim-based comorbidities
    """

    S = SparkData(
        spark=spark,
        patient_type="inpatient",
        dataset_type="all",
    )

    notes = S.orbis_note.join(
        S.visit_occurrence.select(["visit_occurrence_id"]),
        on="visit_occurrence_id",
        how="inner",
    )
    notes = notes.select(
        [
            "person_id",
            "visit_occurrence_id",
            "note_id",
            F.lower(F.col("note_text")).alias("note_text"),
        ]
    ).drop_duplicates()

    notes_charlson = notes.filter(F.col("note_text").rlike("charlson"))

    notes_charlson_pd = notes_charlson.toPandas()

    nlp = get_nlp(no_pipes=True)

    nlp.add_pipe("eds.charlson")

    note_nlp = pipe(
        note=notes_charlson_pd,
        nlp=nlp,
        n_jobs=-1,
        extensions=["score_value", "score_name"],
    )

    kept_notes = spark.createDataFrame(note_nlp[["note_id", "score_value"]])

    kept_visits = (
        S.orbis_note.select(["note_id", "visit_occurrence_id"])
        .join(
            kept_notes,
            on="note_id",
            how="inner",
        )
        .groupby("visit_occurrence_id")
        .agg(F.max("score_value").alias("score_value"))
        .join(
            S.visit_occurrence,
            on="visit_occurrence_id",
            how="inner",
        )
        .cache()
    )

    S.visit_occurrence = kept_visits

    S.get_notes()

    # Adding care site
    spark.sql(f"USE {S.db_name}")
    care_site = spark.sql(
        """
        SELECT
            SUBSTRING(location_cd, 5, 3) AS care_site_id,
            encounter_num AS visit_occurrence_id
        FROM i2b2_observation_ufr
        """
    ).drop_duplicates()

    final_visits = S.visits.join(
        care_site,
        on="visit_occurrence_id",
        how="left",
    )

    final_notes = S.notes.toPandas()
    final_visits = final_visits.toPandas()

    final_visits = final_visits.merge(
        final_notes[
            ["person_id", "visit_occurrence_id", "note_id", "note_text", "score_value"]
        ],
        on=["person_id", "visit_occurrence_id"],
        how="inner",
    )

    final_visits = final_visits.rename(columns=dict(score_value="mentionned_charlson"))

    return final_visits
