from IPython.display import HTML, display
from labeltool.labelling import Labelling

from ecci import DATA_DIR
from ecci.config import COMORB_CONFIG


def get_labelling_tool(patient_type: str = "outpatient"):
    folder_path = DATA_DIR / patient_type

    modifiers_params = [
        {
            "modifier_name": "negation",
            "selection_type": "binary",
        },
        {
            "modifier_name": "family",
            "selection_type": "binary",
        },
        {
            "modifier_name": "hypothesis",
            "selection_type": "binary",
        },
    ]

    params = dict(
        folder_path=folder_path,
        labels_params=COMORB_CONFIG,
        modifiers_params=modifiers_params,
        window_snippet=50,
    )

    if patient_type == "rare_comorbs":
        params["use_stay_labels"] = False
        disable_add_list()

    LabellingTool = Labelling(**params)

    return LabellingTool


def disable_add_list():
    custom_css = """<style>.add_list_class
        {display: none;}
        </style>"""
    display(HTML(custom_css))
