import os

ALL_DB = dict(
    cse180032={
        "USER": "cse180032",
        "DB_TYPE": "i2b2",
        "DB_NAME": "cse_180032_20210713",
        "DATASET_TYPE": "test",
    },
    cse200055={
        "USER": "cse200055",
        "DB_TYPE": "i2b2",
        "DB_NAME": "cse_200055_20211019",
        "DATASET_TYPE": "test",
    },
    cse200093={
        "USER": "cse200093",
        "DB_TYPE": "i2b2",
        "DB_NAME": "cse_200093_20210402",
        "DATASET_TYPE": "test",
    },
)

DB = ALL_DB.get(os.environ.get("USER"))
