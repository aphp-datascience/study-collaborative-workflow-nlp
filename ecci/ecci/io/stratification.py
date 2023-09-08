# -*- coding: utf-8 -*-
from typing import Any, Dict

import pyspark.sql.functions as F
from pyspark.sql.session import SparkSession

from ecci.config import DB
from ecci.io.utils import check_hdfs_file_exists

columns = {
    "omop": {
        "person_table": "person",
        "id": "person_id",
        "birthdate": "birth_datetime",
    },
    "i2b2": {
        "person_table": "i2b2_patient",
        "id": "patient_num",
        "birthdate": "birth_date",
    },
}

SPLIT_RATIO = 10
TEST_THRESHOLD = 3  # All strata > this threshold go in the test
CSE_PROJECTS = ["cse200093", "cse180032", "cse200055"]
N_PROJECTS = len(CSE_PROJECTS)


class Stratify(object):
    def __init__(
        self,
        spark: SparkSession,
        config: Dict[str, Any] = DB,
        limit: int = None,
    ):
        """

        Parameters
        ----------
        spark : SparkSession
        config : Dict[str,Any]
            Configuration dict
        limit : int, optional
            Limit the number of patients (for testing), by default None
        """

        self.spark = spark
        self.sql = spark.sql

        self.user = config["USER"]
        self.db_name = config["DB_NAME"]
        self.db_type = config["DB_TYPE"]

        self.person_table_name = columns[self.db_type]["person_table"]
        self.col_id = columns[self.db_type]["id"]
        self.col_bd = columns[self.db_type]["birthdate"]

        self.split_ratio = SPLIT_RATIO

        self.project_idx = CSE_PROJECTS.index(self.user)

        self.limit = limit

    def compute_hash(self):
        """
        Read the person table of the project
        Filter depending on the project ID
        Adding a hash modulo the split ratio to build various train set and a test set
        """
        df = self.sql(
            f"""
                       SELECT
                           {self.col_id},
                           {self.col_bd}
                       FROM
                           {self.db_name}.{self.person_table_name}
                       """
        )

        if self.limit is not None:
            df = df.limit(self.limit)

        self.ids = (
            df.withColumn("bd_string", F.date_format(df[self.col_bd], "Y-M-d"))
            .withColumn("hash", F.abs(F.hash(F.col("bd_string"))))
            .filter(F.col("hash") % N_PROJECTS == self.project_idx)
            .withColumn("double_hash", F.abs(F.hash(F.col("hash"))))
            .withColumn("stratum", F.col("double_hash") % self.split_ratio)
            .select([self.col_id, "stratum"])
            .withColumnRenamed(self.col_id, "person_id")
        )

    def run(self):
        path = f"clinicai/{self.db_name}_stratification.parquet"

        if check_hdfs_file_exists(path):
            print("Récupération de la stratification stockée...")
            self.ids = self.spark.read.load(path)

        else:
            print("Pas de stratification stockée!")
            answer = input("Voulez-vous calculer la stratification ? [y/n]")
            if answer.lower() == "y":
                self.compute_hash()

                self.ids.write.format("parquet").save(path)
                print("Stratification terminée !")
            else:
                return

    def get_dataset(self, what="train", stratum=None):
        if what == "train":
            if stratum is None:
                return self.ids.filter(F.col("stratum") < TEST_THRESHOLD)
            else:
                return self.ids.filter(F.col("stratum") == stratum)
        elif what == "test":
            return self.ids.filter(F.col("stratum") >= TEST_THRESHOLD)
        elif what == "all":
            return self.ids
