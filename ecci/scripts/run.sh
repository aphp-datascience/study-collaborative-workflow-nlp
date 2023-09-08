#!/bin/bash

set -e

eds-toolbox spark submit --config config.cfg --log-path logs get_notes.py

python get_entities.py
