<div align="center">
<p align="left">
<a href="https://github.com/psf/black" target="_blank">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Black">
</a>
<a href="https://www.python.org/" target="_blank">
    <img src="https://img.shields.io/badge/python-%3E%3D%203.7.10%20%7C%20%3C%3D%203.7.13-brightgreen" alt="Supported Python versions">
</a>
<a href="https://zenodo.org/badge/latestdoi/687513883"><img src="https://zenodo.org/badge/687513883.svg" alt="DOI"></a>
</p>
</div>

# Collaborative and privacy-preserving workflows on a clinical data warehouse: an example developing natural language processing pipelines to detect medical conditions

## Study

This repositoy contains the computer code that has been used for the article:
```
@unpublished{petitjean2023nlp,
author = {Thomas Petit-Jean and Christel Gerardin and Emmanuelle Berthelot and Gilles Chatellier and Marie Frank and Xavier Tannier and Emmanuelle Kempf and Romain Bey},
title = {Collaborative and privacy-preserving workflows on a clinical data warehouse: an example developing natural language processing pipelines to detect medical conditions},
note = {Manuscript submitted for publication},
year = {2023}
}
```
The code has been executed on the database of the <a href="https://eds.aphp.fr/" target="_blank">Greater Paris University Hospitals</a>

:warning:
This repository is not maintained. It contains computer code that is specific to a research study.


## Version 1.0.0
- Code of article after review.

## Repository description

- `edsml`: library developped to facilitate the development of the qualification algorithms. Based on `pytorch` and `pytorch-lightning`.
- `ecci_qualifier`: source code of the machine-learning based qualification algorithm.
- `ecci`: code used during training for document selection, CLAIM pipeline or annotation process.
- `analysis`: code used to generate results for the paper

## Acknowledgement

We would like to thank [Assistance Publique – Hôpitaux de Paris](https://www.aphp.fr/) and [AP-HP Foundation](https://fondationrechercheaphp.fr/) for funding this project.
