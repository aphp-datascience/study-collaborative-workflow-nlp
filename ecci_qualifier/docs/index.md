# Overview

<p align="center">
  <img src="assets/logo.svg" width="30%">
</p>

EDS-Charlson-Qualifier is a project used at APHP to qualify entities.
It is built on top of [edsnlp](https://github.com/aphp/edsnlp) and [edsml](https://gitlab.eds.aphp.fr/datasciencetools/eds-ml).

This model can be used to discard **irrelevant entities** that are:

- Negated
- Hypothetical
- Not related to the patient himself
- Incorrect (e.g. if a disease is extracted inside the mention of a care site's name)

!!! warning
    The model was trained to output a single boolean value to qualify each entity, telling if the entity should be kept or not. It is not able to distinguish between the four modalities described above

## Presentation

The model was developped in parallel with the development of 18 NER pipelines available on EDS-NLP. Those pipelines extracts 16 conditions from the Charlson Comorbidity Index, along with alcohol and tobacco consumption.
The present model was thus developped to qualify those extraction, i.e., to classify them into **relevant** and **irrelevant** entities. It this context, **irrelevant entities** can be

## Installation

```bash
pip install ecci_qualifier
```
