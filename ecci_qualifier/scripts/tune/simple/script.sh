#!/bin/bash

eds-toolbox slurm submit --config ../../slurm.cfg -c "python script.py ../../configs/$1.cfg"
