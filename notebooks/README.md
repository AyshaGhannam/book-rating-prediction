Project notebooks and execution instructions
# Phase 1 Notebooks

See the [main project README](../README.md) for setup, full run instructions, job tests, and results.

| Notebook | Owner | Purpose |
|---|---|---|
| `00_check_project_access` | Aysha | Checks that Databricks can read the raw Goodreads CSV. |
| `01_clean_goodreads` | Aysha | Validates the raw data and publishes the cleaned ratings table. |
| `02_prepare_features` | Nour | Creates reproducible splits and publishes training, validation, and test features. |
| `03_eda_goodreads` | Aysha | Analyzes the cleaned ratings and compares raw CSV and cleaned Parquet storage sizes. |
| `04_verify_ingestion_rerun` | Aysha | Checks that the existing raw CSV is skipped and left unchanged. |

## Run order

- **Automated pipeline:** Run `proj-bookrating-phase1`. The job runs `01_clean_goodreads` and then `02_prepare_features`.
- **Manual pipeline:** Run `01_clean_goodreads` first. Once the cleaned table exists, run `02_prepare_features` for features or `03_eda_goodreads` for analysis. EDA does not need the feature outputs.
- **Separate checks:** Use `00_check_project_access` to check storage access and `04_verify_ingestion_rerun` to verify that the existing raw file stays in place.
