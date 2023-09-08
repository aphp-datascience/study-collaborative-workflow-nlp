from confection import Config
from edstoolbox import SparkApp
from loguru import logger

from ecci import DATA_DIR
from ecci.stats import get_mentionned_charlson

app = SparkApp("ecci-get-mentionned-charlson")


@app.submit
def main(spark, sql, config: Config):
    config = config["mentionned_charlson"]
    if not config["run"]:
        logger.info("Skipping 'mentionned_charlson")
        return

    mentionned_charlson = get_mentionned_charlson(spark)

    mentionned_charlson.to_pickle(DATA_DIR / "mentionned_charlson.pickle")


if __name__ == "__main__":
    app.run()
