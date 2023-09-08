from typing import Optional

import pandas as pd
from edsnlp.processing import pipe
from spacy.language import Language

from ecci.config import COMORB_CONFIG

from .pipeline import get_nlp, get_pipes
from .utils import convert_status


class Extractor(object):
    """
    Class to

    - Extract entities
    - Save a subset of notes to be validated
    """

    def __init__(
        self,
        notes: pd.DataFrame,
        qualifiers: bool = True,
        nlp: Optional[Language] = None,
    ):
        """
        Prepare extraction

        Parameters
        ----------
        notes : pd.DataFrame
            DataFrame containing notes of interest
        qualifiers : bool, optional
            Whether to include rule-based qualification, by default True
        nlp : Language, optional
        """
        self.notes = notes
        self.nlp = nlp or get_nlp(qualifiers=qualifiers)
        self.pipes = get_pipes()
        self.qualifiers = qualifiers

    def run(
        self,
    ):
        """
        Extract comorbidities
        """

        ents = pipe(
            self.notes,
            self.nlp,
            additional_spans=list(self.pipes.keys()),
            extensions=[
                "status",
                "negation",
                "hypothesis",
                "family",
                "to_keep",
            ],
            context=["note_id", "visit_occurrence_id"],
            n_jobs=1,
        )

        self.results = pd.DataFrame(ents)
        self.results["has_rule_based_qualifiers"] = self.qualifiers
        self.results.status = self.results.apply(
            lambda row: convert_status(row.label, row.status),
            axis=1,
        )

        print("N results: ", len(self.results))

        mapping = pd.DataFrame(COMORB_CONFIG)[["pipe_name", "label_name"]]
        self.results = self.results.merge(
            mapping,
            left_on="label",
            right_on="pipe_name",
            how="inner",
            validate="many_to_one",
        ).drop(columns=["label", "pipe_name"])

        self.results.rename(
            columns={
                "status": "label_value",
                "start": "offset_begin",
                "end": "offset_end",
            },
            inplace=True,
        )
        print("N results: ", len(self.results))
