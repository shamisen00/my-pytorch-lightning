#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh
python3 train_cli.py fit --config config1.yaml --trainer.logger false --trainer.max_epochs 1