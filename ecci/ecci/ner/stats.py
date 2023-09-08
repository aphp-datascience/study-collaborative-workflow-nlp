import pandas as pd

from ecci.config import COMORB_CONFIG


class Stats(object):
    """
    Class to

    - Extract stats from entities
    """

    def __init__(
        self,
        notes: pd.DataFrame,
        entities: pd.DataFrame,
    ):
        """
        Prepare extraction

        Parameters
        ----------
        notes : pd.DataFrame
            DataFrame containing notes of interest
        notes : pd.DataFrame
            DataFrame containing extracted entities
        """
        self.notes = notes
        self.entities = entities
        self.config = pd.DataFrame(COMORB_CONFIG)
        self.results = dict()

    def get_n_entities_total(self):
        """
        For each comorbidity, computes the number of corresponding entities
        """
        n_entities = pd.DataFrame(
            index=self.config.label_name,
            data=dict(
                N=0,
            ),
            dtype=int,
        )

        updated_n_entities = self.entities.label_name.value_counts().to_frame()
        updated_n_entities.columns = ["N"]
        updated_n_entities.index.name = "label_name"

        n_entities.update(updated_n_entities)

        n_entities.N = n_entities.N.astype(int)

        return n_entities

    def get_n_entities_per_doc(self):
        """
        For each document, computes how many entities were extracted
        """
        n_entities = pd.DataFrame(
            index=self.notes.note_id,
            data=dict(
                N=0,
            ),
            dtype=int,
        )

        updated_n_entities = self.entities.note_id.value_counts().to_frame()
        updated_n_entities.columns = ["N"]

        n_entities.update(updated_n_entities)
        n_entities.N = n_entities.N.astype(int)
        n_entities.sort_values(by="N", inplace=True)

        n_entities = pd.DataFrame(
            data=dict(
                note_index=list(range(1, len(n_entities) + 1)),
                N=list(n_entities.N),
            )
        ).sort_values(by="N")

        return n_entities

    def run(
        self,
    ):
        """
        Extract statistics
        """

        for func in ["get_n_entities_total", "get_n_entities_per_doc"]:
            self.results[func] = getattr(self, func)()
