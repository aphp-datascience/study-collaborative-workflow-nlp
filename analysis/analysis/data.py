import glob
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
from loguru import logger

from analysis import DATA_DIR as SAVE_DATA_DIR
from analysis.config import CSES, PATIENT_TYPES
from ecci import DATA_DIR, EXPORT_DIR
from ecci.config import COMORB_CONFIG


def get_raw(
    cse: str,
    patient_type: str = "outpatient",
    data_type: Optional[str] = "texts",
) -> pd.DataFrame:
    """
    Helper to get a specific DataFrame from the project's

    Parameters
    ----------
    cse : str
        _description_
    patient_type : str, optional
        "inpatient" or "outpatient"
    data_type : Optional[str], optional
        Name of the DataFrame to retrieve, by default "texts"

    Returns
    -------
    pd.DataFrame
    """

    path = DATA_DIR / cse / "*"
    all_data = sorted(glob.glob(str(path)))

    if not all_data:
        logger.info(f"No data found in {str(path)}")
        return

    # data folders have a date name -> we take the last one

    last = all_data[-1]

    if patient_type is not None:
        file = Path(last) / patient_type / f"{data_type}.pickle"

        if not file.exists():
            logger.info(f"File {str(file)} does not exists")
            return

        logger.info(f"Getting file {str(file)} ...")

        data = pd.read_pickle(file)

        data["cse"] = cse
        data["patient_type"] = patient_type

        return data

    else:
        file = Path(last) / f"{data_type}.pickle"

        if not file.exists():
            logger.info(f"File {str(file)} does not exists")
            return

        logger.info(f"Getting file {str(file)} ...")

        data = pd.read_pickle(file)

        data["cse"] = cse

        return data


def get_all_raw(
    data_type: str = "texts",
    as_dict: bool = False,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Get concatenated DataFrame from the 3 CSE projects' data folder

    Parameters
    ----------
    data_type : str, optional
        The name of the DataFrame to retrieve, by default "texts"
    as_dict : bool, optional
        Wether to return concatenated DataFrame or a per-CSE dictionary

    Returns
    -------
    pd.DataFrame
    """
    assert data_type in {
        "annotated",
        "entities",
        "texts",
        "visits",
        "mentionned_charlson",
    }, f"{data_type} -> Incorrect data"

    results = dict()

    patient_types = (
        [None]
        if data_type
        in {
            "mentionned_charlson",
        }
        else PATIENT_TYPES
    )

    for cse in CSES:
        tmp_results = []
        for patient_type in patient_types:
            result = get_raw(cse, patient_type, data_type)
            tmp_results.append(result)

        results[cse] = pd.concat(tmp_results)

    if as_dict:
        return results

    results = pd.concat([df for df in results.values() if df is not None])
    return results


def get_all_raw_export(
    data_type: str = "stats",
    as_dict: bool = False,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Get concatenated DataFrame from the 3 CSE projects' export folder

    Parameters
    ----------
    data_type : str, optional
        The name of the DataFrame to retrieve, by default "texts"
    as_dict : bool, optional
        Wether to return concatenated DataFrame or a per-CSE dictionary

    Returns
    -------
    pd.DataFrame
    """

    results = dict()

    for cse in CSES:
        path = EXPORT_DIR / cse / f"{data_type}.pickle"
        assert path.exists(), f"Export file {data_type} not found for {cse}"
        result = pd.read_pickle(path)
        if "cse" not in result.columns:
            result["cse"] = cse
        results[cse] = result

    if as_dict:
        return results

    results = pd.concat([df for df in results.values() if df is not None])
    return results


def get_charlson_weights(revision: str = "quan") -> pd.DataFrame:
    """
    Get the per-comorbidity Charlson weights from COMORB_CONFIG

    Parameters
    ----------
    revision : str, optional
        Either
        - `base` (weights from princeps article)
        - `quan` (weights from Quan article)

    Returns
    -------
    pd.DataFrame
    """
    assert revision in {"base", "quan"}

    key = "ecci_weight_quan" if revision == "quan" else "ecci_weight"

    weights = pd.DataFrame(
        [
            {
                "label_name": comorb["label_name"],
                "label_name_updated": comorb.get(
                    "family", comorb["label_name"]
                ),  # grouping malignancy together
                "ecci_weight": comorb[key],
            }
            for comorb in COMORB_CONFIG
        ]
    )

    weights = weights.explode("ecci_weight")

    weights["label_value"] = "1"
    weights.loc[weights.index.duplicated(keep="first"), "label_value"] = "2"

    return weights


def get(
    name: str,
    cse: Optional[str] = None,
    model: Optional[str] = None,
    export: bool = False,
):
    cse = cse or os.environ.get("CSE")
    model = model or os.environ.get("MODEL")

    export = "export" if export else ""

    df_path = SAVE_DATA_DIR / model / cse / export / f"{name}.pickle"

    return pd.read_pickle(df_path)


def save(
    obj: Any,
    name: str,
    description: Optional[str] = None,
    cse: Optional[str] = None,
    model: Optional[str] = None,
    export: bool = False,
    **kwargs,
):
    cse = cse or os.environ.get("CSE")
    model = model or os.environ.get("MODEL")
    export = "export" if export else ""

    parent_path = SAVE_DATA_DIR / model / cse / export
    save_path = parent_path / name

    parent_path.mkdir(parents=True, exist_ok=True)

    description = description or name

    if hasattr(obj, "savefig"):
        save_path = save_path.with_suffix(".png")
        obj.savefig(save_path, **kwargs)

    else:
        save_path = save_path.with_suffix(".pickle")
        pd.to_pickle(obj, save_path, **kwargs)

    description = (
        f"[{model.upper()}] - [{cse.upper()}] {description} saved at {save_path}"
    )

    logger.info(description)
