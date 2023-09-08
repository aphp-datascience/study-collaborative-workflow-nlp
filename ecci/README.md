# Installation

First clone the repository. Then:

```bash
cd scripts/chores
./setup.sh
```

# Usage

In the `script` folder, various getter are available:

| file              | description                                                  |
| ----------------- | ------------------------------------------------------------ |
| `get_icd10.py`    | Reads the appropriate Google Sheet and retrieve ICD-10 codes |
| `get_notes.py`    | Retrieve notes for annotation                                |
| `get_entities.py` | Runs the NER pipeline on the retrieved notes.                |
| `get_stats.py`    | Extracts stats from entities.                                |

The `run.sh` retrieve notes and runs the NER pipeline.

# Labelling

The `labelling.ipynb` is used to pop an instance of `LabelTool`
