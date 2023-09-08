from confection import Config
from edstoolbox import SparkApp

from ecci import DATA_DIR
from ecci.io import SparkData
from ecci.io.utils import to_pickle

app = SparkApp("ecci-get-notes")


@app.submit
def main(spark, sql, config: Config):
    config = config["notes"]
    if not config["run"]:
        return

    patients = {
        patient_type: patient_config
        for patient_type, patient_config in config["patients"].items()
        if patient_config["include"]
    }

    for patient_type, patient_config in patients.items():
        data_getter = SparkData(
            spark,
            patient_type=patient_type,
            keep_last_note_only=patient_config["keep_last_note_only"],
            seed=config["seed"],
        )
        n_notes = patient_config["n_notes"]
        data_getter.get_notes(n_notes=n_notes)

        notes = data_getter.notes.toPandas()
        notes["title"] = notes["note_id"].copy()
        to_pickle(notes, DATA_DIR / patient_config["filename"])

        if patient_type.startswith("in"):
            visits = data_getter.visits.toPandas()
            # icd10 = data_getter.icd10.toPandas()
            to_pickle(visits, DATA_DIR / patient_config["visit_filename"])
            # to_pickle(icd10, DATA_DIR / patient_config["icd10_filename"])


if __name__ == "__main__":
    app.run()
