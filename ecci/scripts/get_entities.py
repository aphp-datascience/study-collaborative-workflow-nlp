import pandas as pd
from confection import Config

from ecci import DATA_DIR
from ecci.io.utils import to_pickle
from ecci.ner import Extractor


def main(config: Config):
    patients = {
        patient_type: patient_config
        for patient_type, patient_config in config["notes"]["patients"].items()
        if patient_config["include"]
    }

    config = config["entities"]
    if not config["run"]:
        return

    for patient_config in patients.values():
        notes_path = DATA_DIR / patient_config["filename"]
        entities_path = DATA_DIR / patient_config["filename"].replace(
            "texts", "entities"
        )

        notes = pd.read_pickle(notes_path)

        print(f"Using {len(notes)} notes from {notes_path}")

        entity_getter = Extractor(
            notes=notes,
            qualifiers=config["use_qualifiers"],
        )
        entity_getter.run()
        entities = entity_getter.results
        to_pickle(entities, entities_path)


if __name__ == "__main__":
    config = Config().from_disk("./config.cfg")
    main(config)
