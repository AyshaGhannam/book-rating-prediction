# Book Rating Prediction

**Course:** DSAI3202 — Phase 1  
**Team:** Aysha Ghannam(60308393) and Nour Ben Ayed(60305269)

## Project goal and hypothesis

This project predicts the 1–5 star rating a reader would give a book. We hypothesize that a Random Forest using reader and book rating histories will achieve a lower test mean absolute error (MAE) than a baseline that predicts the book's average training rating. For unseen books, the baseline uses the overall training average.

Phase 1 prepares and validates the data and features. Model training and evaluation are planned for Phase 2.

## Dataset and architecture

We use the Goodreads interactions dataset from the UCSD Book Graph. The source file is `goodreads_interactions.csv`. A rating of 0 means no explicit rating was given, so those interactions are excluded from the prediction dataset.

**Pipeline:** Goodreads CSV → Azure raw storage → Databricks validation and cleaning → Snappy Parquet in Azure processed storage → Unity Catalog cleaned ratings table. The cleaned table feeds exploratory data analysis and feature preparation. The feature notebook saves the train, validation, and test outputs in Azure curated storage and registers them in Unity Catalog.

## Team contributions

| Contributor | Phase 1 work |
|---|---|
| **Aysha Ghannam** | Set up project storage, uploaded the raw dataset, documented the source in the manifest, and checked that the ingestion guard left the existing raw file unchanged. Cleaned and validated the ratings, published the Parquet output, registered the cleaned table, and completed EDA and storage measurements. Added the `input_path` widget used in the automated failure test. |
| **Nour Ben Ayed** | Created reproducible train, validation, and test splits. Built and validated the four training-only features, published the feature outputs, and registered the three feature tables. Configured the Databricks job and tested successful execution, rerun, controlled failure, and recovery. |

## Aysha Ghannam — Ingestion, cleaning, and EDA

### Data preparation results

| Measure | Result |
|---|---:|
| Raw interactions | 228,648,342 |
| Unrated interactions excluded | 124,096,793 |
| Explicit ratings saved | 104,551,549 |
| Invalid field values found | 0 |
| Repeated reader–book pairs found | 0 |

The cleaned table is `dsai3202_databricks.proj_bookrating.bookrating_ratings_v1`. It contains one row per reader–book pair. The `rating` column contains explicit ratings from 1 to 5 and is the prediction target.

The raw CSV occupied 4.319 GB. The cleaned output occupied 0.490 GB across 33 Snappy-compressed Parquet files, an 88.65% raw-to-clean size reduction. This reflects both the removal of unrated rows and the change in file format.

### Ingestion and cleaning checks

- The ingestion guard found the existing Goodreads v1 CSV and skipped the upload. The file path, size, and modification time stayed unchanged. See [ingestion rerun evidence](evidence/ingestion_rerun.txt).
- The cleaning rerun reused the published data and report. The saved output still contained 104,551,549 ratings, with the expected column names and types.
- A separate notebook-level test rejected input missing `rating` and left the published files unchanged.

### Exploratory data analysis

The full cleaned dataset contains 104,551,549 explicit ratings from 816,371 readers across 2,325,541 books. The mean rating was 3.939 stars, and 69.83% of ratings were 4 or 5 stars.

The approximate median was 55 ratings per reader and 4 ratings per book. About 51.34% of books and 8.09% of readers had fewer than five ratings. An average based on a short history can be unstable, so the planned model uses rating counts alongside averages.

The Pearson correlation between `log(1 + book rating count)` and average book rating was 0.0219, indicating little linear association on that scale. These results describe the full cleaned dataset. The model features are computed from training ratings only.

### How to run Aysha's notebooks

1. Open the `dsai3202-databricks` course workspace. Attach the notebooks to the course lab cluster and confirm access to project storage and the `dsai3202_databricks.proj_bookrating` schema.
2. Run [`04_verify_ingestion_rerun`](notebooks/04_verify_ingestion_rerun.py) to check that the existing Goodreads v1 CSV is left in place. This check assumes that the raw CSV is already uploaded.
3. Run [`01_clean_goodreads`](notebooks/01_clean_goodreads%20%281%29.py) from top to bottom with the default `input_path`. The notebook validates the raw data, keeps ratings from 1 to 5, and publishes or reuses the cleaned table.
4. Run [`03_eda_goodreads`](notebooks/03_eda_goodreads.py) after the cleaned table is available. It reads the cleaned table independently of the feature notebook.

## Nour Ben Ayed — Splits, features, and automated job

### Reproducible splits

The feature notebook reads Aysha's cleaned ratings table and hashes `user_id` and `book_id` into 100 buckets. Buckets 0–79 are used for training, 80–89 for validation, and 90–99 for testing. The same reader–book pair is assigned to the same split on a rerun.

| Split | Rows |
|---|---:|
| Train | 83,638,178 |
| Validation | 10,457,550 |
| Test | 10,455,821 |
| **Total** | **104,551,549** |

The split counts add up to the cleaned input count, and no reader–book pair appeared in more than one split.

### Feature definitions and leakage prevention

All four features are derived from the training split.

| Feature | Type | Definition |
|---|---|---|
| `reader_avg_rating` | double | Average rating from the reader's eligible training history. |
| `reader_rating_count` | long | Number of ratings in the reader's eligible training history. |
| `book_avg_rating` | double | Average rating from the book's eligible training history. |
| `book_rating_count` | long | Number of ratings in the book's eligible training history. |

For a training row, the notebook subtracts the row's own rating from the reader and book sums and subtracts 1 from their counts before computing the averages. This prevents the target from being included in its own features. Validation and test rows use the complete training aggregates.

If removing a training row leaves no reader or book history, the corresponding count stays at 0 and the average falls back to `(S - rating) / (N - 1)`, where `S` is the sum of all training ratings and `N` is the number of training rows. This excludes the row's own rating from the fallback too. For validation and test rows with no corresponding training history, the fallback is the full training average `S / N`, with a count of 0. At prediction time, the same full training aggregates and unseen-history fallback can be used.

The output keeps `user_id`, `book_id`, `rating` as the target, and `split` alongside the four feature columns.

### Feature validation and publication

Nour tested target exclusion on a small example. Before publishing, the notebook checks that:

- Features have no missing or non-finite values.
- Average ratings are between 1 and 5, and counts are nonnegative.
- Split counts match expectations and no rows are lost or duplicated.

The notebook writes the feature outputs to staging and validates them before replacing the published folders. A failure before publication leaves the previous outputs in place. A normal rerun replaces the existing feature files rather than appending duplicates.

### Published feature tables

All three tables are in `dsai3202_databricks.proj_bookrating`. The storage path prefix is `abfss://curated@bookrating60308393.dfs.core.windows.net/goodreads/v1/features/`.

| Table | Folder under the path prefix |
|---|---|
| `features_train` | `train/` |
| `features_validation` | `validation/` |
| `features_test` | `test/` |

### Automated job and test results

The Databricks job is `proj-bookrating-phase1` (ID `741628860534238`). The `features` task depends on `clean` and runs only after `clean` succeeds.

| Test | Run ID | Duration | Observed result |
|---|---|---|---|
| Successful run | `692763854509075` | 9m 35s | Both tasks succeeded. |
| Rerun | `79041621458263` | 9m 30s | Both tasks succeeded, counts matched, and no duplicate rows were added. |
| Missing `rating` column | `993624055488619` | 26s | `clean` rejected the input before publication. `features` was skipped, and the published tables stayed unchanged. |
| Normal run after failure | `34636616976155` | 9m 34s | Both tasks succeeded and reported the expected counts. |

For the controlled failure, Nour overrode the `input_path` parameter for that run and pointed it to a separate copy of [`fixture_missing_rating.csv`](tests/fixture_missing_rating.csv) accessible to the job. The recovery run used the normal Goodreads input again.

### How to run the job

1. In `dsai3202-databricks`, start the course lab cluster configured for the job, `dsai3202-lab5-60305269`.
2. Open **Jobs & Pipelines → proj-bookrating-phase1 → Run now**. Keep the default `input_path` for a normal run.
3. After both tasks succeed, verify the published row counts:

```sql
REFRESH TABLE dsai3202_databricks.proj_bookrating.features_train;
REFRESH TABLE dsai3202_databricks.proj_bookrating.features_validation;
REFRESH TABLE dsai3202_databricks.proj_bookrating.features_test;

SELECT 'train' AS split, COUNT(*) AS n
FROM dsai3202_databricks.proj_bookrating.features_train
UNION ALL
SELECT 'validation', COUNT(*)
FROM dsai3202_databricks.proj_bookrating.features_validation
UNION ALL
SELECT 'test', COUNT(*)
FROM dsai3202_databricks.proj_bookrating.features_test;
```

The expected counts are 83,638,178 for training, 10,457,550 for validation, and 10,455,821 for testing. To reproduce the controlled failure, override `input_path` for one run with the full Azure path to the separate missing-rating fixture.

### Resource use and cleanup

The normal job took about 9.5 minutes: roughly 6.5 minutes for cleaning and 3 minutes for feature preparation. It ran on Databricks Runtime 18 LTS with two workers, each with 4 cores and 16 GB of memory.

Checking the staged feature files before publishing adds work to each run, but lets validation catch problems before the published folders are replaced. After collecting run evidence, stop the lab cluster when neither teammate is using it. Retain the raw snapshot, published outputs, and run evidence.

### Nour's files and reused code

- [Feature notebook](notebooks/02_prepare_features.py)
- [Exported job configuration](jobs/proj-booking-phase1.json)
- [Missing-rating fixture](tests/fixture_missing_rating.csv)

Nour reported no reused code in her part.

## AI Use

| Contributor | AI tool | What I asked | What it returned | What I changed or verified |
|---|---|---|---|---|
| Aysha Ghannam | ChatGPT | Help writing and debugging the PySpark and SQL cleaning code. | Drafted validation, duplicate checks, rating filters, Parquet export, and table registration code. Suggested a fix for the unsupported write mode. | Applied the fix and ran the notebook in Databricks. Verified 104,551,549 cleaned ratings and confirmed that the rerun reused the existing data and report. |
| Aysha Ghannam | ChatGPT | Help writing a missing-column test and checking the raw ingestion rerun guard. | Drafted the validation test and a check comparing raw file metadata before and after skipping the upload. | Verified that the test rejected input missing `rating` and left the published files unchanged. Confirmed that the raw file's path, size, and modification time stayed unchanged. |
| Aysha Ghannam | ChatGPT | Help writing the exploratory data analysis code. | Drafted code for rating distributions, reader and book activity, correlation, and storage measurements. | Ran the analysis on the full cleaned dataset and checked the outputs. Interpreted the 88.65% reduction as the combined effect of filtering and changing the file format. |
| Aysha Ghannam | ChatGPT | Help adding a configurable input path for the failure test. | Suggested an `input_path` widget that defaults to the Goodreads CSV. | Added the widget so the job could use a separate CSV for the controlled failure. |
| Aysha Ghannam | ChatGPT | Help combining both contributors' README sections and checking the training fallback. | Organized the shared README, clarified the fallback from the feature code, and identified filename mismatches in links. | Used the combined draft to document each contributor's work separately and describe the fallback implemented in the feature notebook. |
| Nour Ben Ayed | Claude | Help planning Phase 1 and writing the split and feature code. | Drafted the hash split and four training-only features, with each training row's own rating removed from its history features. | Ran the notebook in Databricks, verified the split totals and absence of overlap, and tested target exclusion on a small example. |
| Nour Ben Ayed | Claude | Help writing feature checks, staging, publication, and table registration. | Drafted checks for missing and non-finite values, rating ranges, nonnegative counts, and row totals, plus staging and table registration steps. | Ran the checks, found zero bad rows, and verified the counts in all three published tables. |
| Nour Ben Ayed | Claude | Help setting up the job and testing failure using Aysha's `input_path` widget. | Suggested job setup steps, the parameter override, and a CSV fixture without `rating`. | Ran the job four times and verified the successful run, unchanged rerun counts, skipped feature task on bad input, and successful recovery. |
