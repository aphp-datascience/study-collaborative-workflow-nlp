from typing import Optional, Union

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col, row_number
from pyspark.sql.session import SparkSession
from pyspark.sql.window import Window

from ecci import BASE_DIR
from ecci.config import DB
from ecci.icd10 import list_to_regex
from ecci.io.stratification import Stratify

TABLES_MAPPING = {
    "i2b2_observation_doc": "orbis_note",
    "i2b2_patient": "person",
    "i2b2_visit": "visit_occurrence",
    "i2b2_observation_cim10": "condition_occurrence",
}

I2B2_RENAMING = {
    "i2b2_observation_doc": {
        "concept_cd": "note_class_source_value",
        "encounter_num": "visit_occurrence_id",
        "patient_num": "person_id",
        "instance_num": "note_id",
        "observation_blob": "note_text",
        "start_date": "note_datetime",
    },
    "i2b2_patient": {
        "patient_num": "person_id",
        "birth_date": "birth_datetime",
        "sex_cd": "gender_source_value",
    },
    "i2b2_visit": {
        "patient_num": "person_id",
        "encounter_num": "visit_occurrence_id",
        "start_date": "visit_start_datetime",
        "end_date": "visit_end_datetime",
        "type_visite": "visit_source_value",
    },
    "i2b2_observation_cim10": {
        "encounter_num": "visit_occurrence_id",
        "patient_num": "person_id",
        "instance_num": "condition_occurrence_id",
        "tval_char": "condition_status_source_value",
        "concept_cd": "condition_source_value",
    },
}

OMOP_RENAMING = {
    omop_table: {v: v for k, v in I2B2_RENAMING[i2b2_table].items()}
    for i2b2_table, omop_table in TABLES_MAPPING.items()
}

RENAMING = {"i2b2": I2B2_RENAMING, "omop": OMOP_RENAMING}


class SparkData(object):
    """
    Class used to
    - Access data from the Hive cluster
    - Stratify data and access the stratified data
    - Get notes and ICD-10
    """

    def __init__(
        self,
        spark: SparkSession,
        max_number_patients: Union[int, None] = None,
        patient_type: str = "inpatient",
        dataset_type: Optional[str] = None,
        keep_last_note_only: bool = True,
        seed: int = 42,
    ):
        self.spark = spark
        self.seed = seed

        assert patient_type in {"inpatient", "outpatient"}
        if patient_type == "inpatient":
            self.visit_type = "Hosp"
            self.note_type = "CR:CRH-HOSPI"
        else:
            self.visit_type = "Ext"
            self.note_type = "CR:CR-CONS"

        self.patient_type = patient_type
        self.keep_last_note_only = keep_last_note_only

        # Getting parameters from the config file
        self.db_name = DB["DB_NAME"]
        self.db_type = DB["DB_TYPE"]
        self.dataset_type = dataset_type or DB["DATASET_TYPE"]
        self.stratum = DB.get("LAST_STRATUM", None)

        # Data stratification
        self.Stratify = Stratify(spark)
        self.Stratify.run()

        # Getting data
        self.max_number_patients = max_number_patients
        self._get_data()

    def to_omop(
        self,
        table: str,
        ids: Union[DataFrame, None] = None,
    ) -> DataFrame:
        """
        Read an I2B2 or OMOP table, and returns it in OMOP format.
        If provided, restricts to ids present in the `ids` DataFrame

        Parameters
        ----------
        table: str
            The table name
        ids: DataFrame, optionnal
            A Spark DataFrame with a column `person_id`
            If provided, an inner join is done before returning the loaded table

        Returns
        -------
        df: DataFrame
            The loaded table
        """

        columns = RENAMING[self.db_type][table]
        query = ",".join(["{k} AS {v}".format(k=k, v=v) for k, v in columns.items()])

        df = self.spark.sql(
            f"""SELECT
                    {query}
                FROM
                    {self.db_name}.{table}"""
        )

        if table == "i2b2_observation_cim10":
            # Removing the `CIM10:` prefix

            df = df.withColumn(
                "condition_source_value",
                F.substring(F.col("condition_source_value"), 7, 20),
            )

        if table == "i2b2_visit":
            visit_type_mapping = {
                "I": "Hosp",
                "II": "Incomp",
                "U": "Urg",
                "O": "Ext",
            }
            df = df.replace(visit_type_mapping, subset=["visit_source_value"])
            df = df.filter(F.col("visit_source_value") == self.visit_type)

        if ids is not None:
            df = df.join(
                ids.select("person_id"),
                on="person_id",
                how="inner",
            )

        return df

    def _get_data(self):
        """
        Loading the needed tables.
        Joining each of them with the stratification's output
        """

        ids = None

        if self.dataset_type != "all":
            ids = self.Stratify.get_dataset(
                what=self.dataset_type,
                stratum=self.stratum,
            )

            if self.max_number_patients is not None:
                print(
                    f"Restricting the dataset to ~ {self.max_number_patients} patients"
                )
                fraction = self.max_number_patients / ids.count()
                ids = ids.sample(fraction, seed=self.seed).cache()

        if self.db_type == "i2b2":
            tables = TABLES_MAPPING.keys()
        elif self.db_type == "omop":
            tables = TABLES_MAPPING.values()

        attributes = TABLES_MAPPING.values()

        for attribute, table in zip(attributes, tables):
            setattr(self, attribute, self.to_omop(table, ids=ids))

    def _add_cim10_comorb(
        self,
        via_regex: bool = False,
    ):
        """
        Create a `visit_comorbs` DataFrame and stores it as an attribute.
        - Lines: visit occurrences
        - Columns:
           For each `comorb`, a `cim10_{comorb}` is created, containing
           the value of the phenotyping extracted from CIM10.
           For comorbidities with various levels of seriousness,
           the most severe ont is kept if multiples are found
        """

        cim = self.condition_occurrence

        codes_path = BASE_DIR / "icd10" / "icd10.pickle"

        if not codes_path.exists():
            raise FileNotFoundError("ICD10 codes not found. Check the ICDGetter class")

        codes = pd.read_pickle(codes_path)

        for comorb, comorb_df in codes.groupby("comorb"):
            if via_regex:
                main_codes = comorb_df[comorb_df["Hiérarchie"] == "Principal"]
                print(f"Number of codes for {comorb}: {len(main_codes)}")

                cond = F.when(F.lit(False), F.lit(None))
                for status, comorb_status_df in main_codes.groupby("Status"):
                    pattern = list_to_regex(list(comorb_status_df.code))
                    cond = cond.when(
                        F.col("condition_source_value").rlike(pattern),
                        F.lit(status),
                    )

            else:
                cond = F.when(F.lit(False), F.lit(None))
                for status, comorb_status_df in comorb_df.groupby("Status"):
                    cond = cond.when(
                        F.col("condition_source_value").isin(
                            list(comorb_status_df.code)
                        ),
                        F.lit(status),
                    )

            cond.otherwise("0:Absence")

            cim = cim.withColumn(f"cim10_{comorb}", cond).fillna("0:Absence")

        cim10_comorbs = [c for c in cim.columns if c.startswith("cim10")]
        self.cim10_comorbs = cim10_comorbs

        visit_comorbs = (
            cim.select(["visit_occurrence_id"] + cim10_comorbs)
            .groupby("visit_occurrence_id")
            .agg(*[F.max(c).alias(c) for c in cim10_comorbs])
        )

        self.visits = self.visits.join(
            visit_comorbs, on="visit_occurrence_id", how="left"
        )
        # Using all previous visits for each selected visit:

        vo = self.visit_occurrence.selectExpr(
            [col + " as all_" + col for col in self.visit_occurrence.columns]
        ).select(
            ["all_person_id", "all_visit_occurrence_id", "all_visit_start_datetime"]
        )

        previous_visits = (
            self.visits.join(
                vo,
                on=(self.visits.person_id == vo.all_person_id)
                & (self.visits.visit_start_datetime >= vo.all_visit_start_datetime),
                how="inner",
            )
            .withColumnRenamed("visit_occurrence_id", "original_visit_occurrence_id")
            .withColumnRenamed("all_visit_occurrence_id", "visit_occurrence_id")
        ).select(["visit_occurrence_id", "original_visit_occurrence_id"])

        previous_cim = (
            cim.join(previous_visits, on="visit_occurrence_id", how="inner")
            .drop("visit_occurrence_id")
            .withColumnRenamed("original_visit_occurrence_id", "visit_occurrence_id")
        )

        previous_visit_comorbs = (
            previous_cim.select(["visit_occurrence_id"] + cim10_comorbs)
            .selectExpr(
                ["visit_occurrence_id"]
                + [col + " as all_" + col for col in cim10_comorbs]
            )
            .groupby("visit_occurrence_id")
            .agg(
                *[
                    F.max(c).alias(c)
                    for c in [f"all_{col_name}" for col_name in cim10_comorbs]
                ]
            )
        )

        self.visits = self.visits.join(
            previous_visit_comorbs,
            on="visit_occurrence_id",
            how="left",
        )

        self.visits_pd = self.visits.toPandas()

    def get_notes(self, n_notes: Optional[int] = None):
        """
        Get notes from current stratum

        Parameters
        ----------
        patient_type : str, optional
            Either 'inpatient' or 'outpatient'
        keep_last_only : bool, optional
            Whether to only keep the last document of each visit


        Returns
        -------
        notes : DataFrame
            Pandas DataFrame with additionnal CIM10 informations
            coming from the corresponding visits
        """

        visits = self.visit_occurrence

        if self.patient_type == "inpatient":
            visits = visits.join(
                self.condition_occurrence.select(
                    ["visit_occurrence_id"]
                ).drop_duplicates(),
                on="visit_occurrence_id",
                how="inner",
            )  # Join with CO to ensure the visit has ICD-10 codes

        notes = self.orbis_note.filter(
            F.col("note_class_source_value") == self.note_type
        ).join(
            visits,
            on=["person_id", "visit_occurrence_id"],
            how="inner",
        )

        notes = notes.filter(F.length("note_text") > 500)

        if self.keep_last_note_only:
            w = Window.partitionBy("visit_occurrence_id").orderBy(
                col("note_datetime").desc()
            )
            notes = (
                notes.withColumn("row", row_number().over(w))
                .filter(col("row") == 1)
                .drop("row")
            )

        self.notes = notes
        self.visits = notes.select(
            ["person_id", "visit_occurrence_id", "visit_start_datetime"]
        ).drop_duplicates()
        # self.notes_pd = notes.toPandas()

        if n_notes is not None:
            self.notes = self.notes.orderBy(F.rand(seed=self.seed)).limit(n_notes)
            self.visits = self.visits.join(
                self.notes.select(["visit_occurrence_id"]),
                on="visit_occurrence_id",
                how="inner",
            )

        if self.patient_type == "inpatient":
            print("Adding ICD10 comorbidities in `self.visits`")
            self._add_cim10_comorb(via_regex=True)

        return
