from confection import Config
from edstoolbox import SparkApp
from loguru import logger

from ecci import DATA_DIR
from ecci.stats import get_validation_dataset

app = SparkApp("ecci-get-validation-dataset")


@app.submit
def main(spark, sql, config: Config):
    config = config["validation_dataset"]
    if not config["run"]:
        logger.info("Skipping validation_dataset")
        return

    for patient_type in config["patient_type"]:
        logger.info(f"Getting dataset for {patient_type}")
        validation_dataset = get_validation_dataset(
            spark,
            patient_type,
            fetch_notes=config["fetch_notes"],
        )

        validation_dataset.to_pickle(
            DATA_DIR / f"validation_dataset_{patient_type}.pickle"
        )


if __name__ == "__main__":
    app.run()
