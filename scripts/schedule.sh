#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh
python3 src/train.py data.num_workers=20 trainer.max_epochs=5 logger=csv

python3 src/train.py data.num_workers=20 trainer.max_epochs=5 debug=profiler

python3 src/train.py data.num_workers=2 trainer.max_epochs=5 logger=mlflow data=pictures model=pictures callbacks.early_stopping.monitor="val/loss" callbacks.model_checkpoint.monitor="val/loss" trainer.accelerator=gpu trainer.devices=1

python3 cli.py fit
