#!/bin/bash

set -e

#eds-toolbox spark submit --config config.cfg --log-path logs get_cohort_stats.py
#eds-toolbox spark submit --config config.cfg --log-path logs get_mentionned_charlson.py
# eds-toolbox spark submit --config config.cfg --log-path logs get_validation_dataset.py
eds-toolbox spark submit --config config.cfg --log-path logs prepare_validation_dataset.py
