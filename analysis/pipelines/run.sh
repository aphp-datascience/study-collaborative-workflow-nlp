#!/bin/bash

set -e

for CSE in cse180032 cse200055 cse200093 Overall
do
    for MODEL in eds base
    do
        export CSE=$CSE && export MODEL=$MODEL && python get_stats.py --step 2
    done
    export CSE=$CSE && export MODEL=base && python get_stats_comparing_models.py
done

export CSE='Overall' && export MODEL='compare'
python get_stats_collaborative.py
python mentionned_charlson_comparison.py

python generate_paper_data.py
cd /export/home/tpetitjean/ecci/analysis/data
python -m zipfile -c paper_data.zip paper_data/
git add -f paper_data.zip
git commit -m "data update"
git push
