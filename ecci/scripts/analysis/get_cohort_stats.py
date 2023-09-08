import os

from confection import Config
from edstoolbox import SparkApp
from loguru import logger

from ecci import EXPORT_DIR
from ecci.stats import get_cohort_stats

app = SparkApp("ecci-get-cohort-stats")


@app.submit
def main(spark, sql, config: Config):
    config = config["cohort_stats"]
    if not config["run"]:
        logger.info("Skipping 'cohort_stats")
        return

    cse = os.environ["USER"]

    cohort_stats = get_cohort_stats(spark)

    cohort_stats.to_pickle(EXPORT_DIR / cse / "stats.pickle")


if __name__ == "__main__":
    app.run()
