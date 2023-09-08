import pandas as pd
import requests

from ecci import BASE_DIR
from ecci.config import COMORB_CONFIG


class ICD10Getter:
    def __init__(
        self,
        google_sheet_id="1GFGR7UqAhAUqSXPWGPEto2IY2fxIKjji",
        save_path=BASE_DIR / "icd10",
    ):
        self.google_sheet_id = google_sheet_id
        self.save_path = save_path

    def get(self):
        """
        Download the collaborative spredsheet and save it
        as an Excel file and as a pickled DataFrame
        """

        URL = "https://docs.google.com/uc?export=download"

        session = requests.Session()

        response = session.get(URL, params={"id": self.google_sheet_id}, stream=True)
        token = get_confirm_token(response)

        if token:
            params = {"id": id, "confirm": token}
            response = session.get(URL, params=params, stream=True)

        save_response_content(response, self.save_path / "icd10.xlsx")

        self.to_pandas()

    def to_pandas(self):
        codes = pd.read_excel(self.save_path / "icd10.xlsx", sheet_name=None)

        all = []
        for comorb in COMORB_CONFIG:
            pipe_name = comorb["pipe_name"]
            sheet_names = comorb["excel_index"]
            sheet_names = (
                sheet_names if isinstance(sheet_names, list) else [sheet_names]
            )
            sheet_names = [str(name) for name in sheet_names]

            for sheet_name in sheet_names:
                comorb_codes = codes[sheet_name].dropna(subset=["code"])
                comorb_codes["comorb"] = pipe_name
                all.append(
                    comorb_codes[["code", "Libellé", "Status", "comorb", "Hiérarchie"]]
                )

        codes = pd.concat(all)
        codes.to_pickle(self.save_path / "icd10.pickle")


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
