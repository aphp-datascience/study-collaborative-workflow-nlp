#!/bin/bash

eds-toolbox slurm submit --config ../slurm.cfg -c "python script.py --config ../configs/$1.cfg"
