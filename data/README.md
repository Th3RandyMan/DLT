# Data Directory

This directory contains all data files for the project:

- `raw/`: Original, immutable data dump
- `processed/`: Cleaned and transformed data ready for analysis
- `external/`: External datasets downloaded from third-party sources

## Data Organization Guidelines

1. Never commit large data files to version control
2. Use `.gitignore` to exclude data files
3. Document data sources and preprocessing steps
4. Keep raw data immutable - always create processed versions