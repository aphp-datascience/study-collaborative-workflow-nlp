from functools import lru_cache

from ecci.config import COMORB_CONFIG

status_mapping = {
    comorb["pipe_name"]: comorb["status_mapping"] for comorb in COMORB_CONFIG
}


@lru_cache(maxsize=None)
def convert_status(
    comorbidity: str,
    status: int,
) -> str:
    """
    Converts an interger-type comorbidity status (from `ent._.status`)
    to a human-readable status

    Parameters
    ----------
    comorbidity : str
        The `pipe_name` of the comorbidity
    status : int
        The status

    Returns
    -------
    str
        The string status
    """
    return status_mapping[comorbidity].get(
        status,
        status_mapping[comorbidity][
            1
        ],  # Handle edge case: "transplantation foierein" -> CKD status to 2
    )
