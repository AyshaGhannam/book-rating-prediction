# Phase 1 - Nour's part

## What I did
Aicha cleaned the Goodreads data. My part was to split it into train, validation and test, make 4 features from it, save them, and test the automated job.

The input is the table `dsai3202_databricks.proj_bookrating.bookrating_ratings_v1` (104,551,549 rows).

## Splits
I used a hash of `user_id` and `book_id` and put each row in one of 100 buckets: 0-79 is train, 80-89 is validation, 90-99 is test. I did it this way so the same pair always ends up in the same split, even if I rerun it.

| Split | Rows |
|---|---|
| train | 83,638,178 |
| validation | 10,457,550 |
| test | 10,455,821 |
| total | 104,551,549 |

The splits add up to the full table, and I checked that no reader-book pair is in more than one split.

## Features
I made 4 features. They only use the training ratings.

| Feature | What it is | Type | Known at prediction time? |
|---|---|---|---|
| reader_avg_rating | the reader's average training rating | double | Yes, from the reader's training history (overall average if the reader is new) |
| reader_rating_count | how many training ratings the reader has | long | Yes (0 if new) |
| book_avg_rating | the book's average training rating | double | Yes, from the book's training history (overall average if the book is new) |
| book_rating_count | how many training ratings the book has | long | Yes (0 if new) |

The source for all four is the training split of the table.

**Leakage:** for a training row I take away its own rating (and 1 from the count) before I calculate its features. If I didn't, the answer would be inside its own features. I tested this on a small example I made by hand and it worked. Validation and test rows use the full training numbers. If a reader or book has no training ratings, I use the overall training average and a count of 0.

The output keeps `user_id`, `book_id`, `rating` (the target) and `split`.

## Checks before saving
- no missing or infinite values in the features
- averages are between 1 and 5 and counts are 0 or more
- no rows lost or duplicated 
- I write to a staging folder first, check it, and only then publish

## Saved tables (v1)
| Table | Path |
|---|---|
| dsai3202_databricks.proj_bookrating.features_train | abfss://curated@bookrating60308393.dfs.core.windows.net/goodreads/v1/features/train/ |
| dsai3202_databricks.proj_bookrating.features_validation | abfss://curated@bookrating60308393.dfs.core.windows.net/goodreads/v1/features/validation/ |
| dsai3202_databricks.proj_bookrating.features_test | abfss://curated@bookrating60308393.dfs.core.windows.net/goodreads/v1/features/test/ |

## Job tests
The job is `proj-bookrating-phase1` (ID 741628860534238). First `clean` runs, then `features`, and `features` only runs if `clean` succeeded.

| Test | Run ID | Duration | Result |
|---|---|---|---|
| Successful run | 692763854509075 | 9m 35s (`clean` 6m 35s, `features` 2m 59s) | both tasks succeeded |
| Rerun | 79041621458263 | 9m 30s (`clean` 6m 33s, `features` 2m 56s) | succeeded, same counts, nothing duplicated |
| Bad input (CSV without `rating`) | 993624055488619 | 26s | `clean` failed with `Unexpected columns: ['user_id', 'book_id', 'is_read', 'is_reviewed']`, `features` was skipped (upstream failed), and the tables stayed the same |
| Normal run after the failure | 34636616976155 | 9m 34s (`clean` 6m 18s, `features` 3m 15s) | succeeded, same counts |

For the bad input test I only changed the `input_path` parameter for that one run (the widget Aicha added) and pointed it to `fixture_missing_rating.csv`.

## How to run it
1. Open Databricks (`dsai3202-databricks`) and start the lab cluster `dsai3202-lab5-60305269`.
2. Go to Jobs & Pipelines, open `proj-bookrating-phase1` and click Run now.
3. When it is done, check the counts:
```sql
REFRESH TABLE dsai3202_databricks.proj_bookrating.features_train;
REFRESH TABLE dsai3202_databricks.proj_bookrating.features_validation;
REFRESH TABLE dsai3202_databricks.proj_bookrating.features_test;

SELECT 'train' AS split, COUNT(*) AS n FROM dsai3202_databricks.proj_bookrating.features_train
UNION ALL SELECT 'validation', COUNT(*) FROM dsai3202_databricks.proj_bookrating.features_validation
UNION ALL SELECT 'test', COUNT(*) FROM dsai3202_databricks.proj_bookrating.features_test;
```
It should give 83,638,178 / 10,457,550 / 10,455,821.


## Resource use
The whole job takes about 9.5 minutes (`clean` about 6.5 minutes and `features` about 3 minutes). Cluster: dsai3202-lab5-60305269, 2 workers (4 cores and 16 GB each), Databricks Runtime 18 LTS.

One choice I made: I check the features in a staging folder before I publish them. It makes the run a bit longer, but if a check fails the published tables are never changed.

## Files
- `notebooks/02_prepare_features.py`
- `jobs/proj-bookrating-phase1.json`
- `tests/fixture_missing_rating.csv`

## Reused code
None

## AI use
| Contributor | AI tool | What I asked | What it returned | What I changed or verified |
|---|---|---|---|---|
| Nour Ben Ayed | Claude | Help planning the Phase 1 steps and writing the split and feature code. | Drafted the hash split (100 buckets) and the four training-only features, with each training row's own rating removed from its history features. | I ran the notebook in Databricks and verified the three splits add up to 104,551,549 rows with no pair in more than one split. I tested a small hand-made example and confirmed each training row's own rating is excluded from its history features. |
| Nour Ben Ayed | Claude | Help writing the feature checks and the publish and table registration step. | Drafted checks for missing or non-finite values, averages between 1 and 5, non-negative counts, row counts, a staging folder, and the three table registrations. | I ran the checks (0 bad rows) and verified the tables contain 83,638,178, 10,457,550 and 10,455,821 rows. |
| Nour Ben Ayed | Claude | Help setting up the job and the failure test using Aicha's `input_path` widget. | Gave the job setup steps, the `input_path` job parameter and the test CSV without `rating`. | I created and ran the job four times. I verified the rerun gave the same counts, `clean` failed on the bad input, `features` was skipped, and the published counts stayed the same.

