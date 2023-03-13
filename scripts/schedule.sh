#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh
python3 src/train.py data.num_workers=20 trainer.max_epochs=5 logger=csv

python3 src/train.py data.num_workers=20 trainer.max_epochs=5 debug=profiler