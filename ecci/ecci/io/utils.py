import pickle
from pathlib import Path
from subprocess import Popen
from typing import Union

import pandas as pd


def check_hdfs_file_exists(path):
    process = Popen(f"hadoop fs -test -e '{path}'", shell=True)
    returncode = process.wait()

    return returncode == 0


def to_pickle(data: pd.DataFrame, path: Union[str, Path]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # data.to_pickle(path)
    with open(path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(path: Union[str, Path]):
    with open(path, "rb") as handle:
        return pickle.load(handle)
