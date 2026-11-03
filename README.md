# book-rating-prediction
## Project goal and hypothesis

This project predicts the 1–5 star rating a reader would give a book.
We hypothesize that a Random Forest using reader and book rating
histories will achieve a lower test MAE than a baseline that predicts
the book's average training rating. For unseen books, the baseline
uses the overall training average.

Phase 1 prepares and checks the data. Model training and evaluation
are planned for a later phase.

## Dataset and architecture

We use the Goodreads interactions dataset from the UCSD Book Graph.
The source file is `goodreads_interactions.csv`. A rating of 0 means
no explicit rating was given, so those rows are excluded from the
prediction dataset.

Pipeline: Goodreads CSV → Azure raw storage → Databricks validation
and cleaning → Snappy Parquet in Azure processed storage → Unity
Catalog table → EDA and feature preparation.

Aysha completed ingestion checks, cleaning, table registration, and
EDA. Nour will use the cleaned table to prepare the splits and
features and demonstrate the automated job.
### Phase 1 data preparation results

The Goodreads v1 source contained 228,648,342 interactions. The cleaning
checks found no invalid field values or repeated user–book pairs. We
excluded 124,096,793 unrated interactions (rating 0) and saved
104,551,549 explicit ratings (1–5 stars).

The cleaned table is
`dsai3202_databricks.proj_bookrating.bookrating_ratings_v1`.
It contains one row per user–book pair. The `rating` column is the target
for the later prediction phase.

The raw CSV occupied 4.319 GB. The cleaned output occupied 0.490 GB
across 33 Snappy-compressed Parquet files, an 88.65% raw-to-clean size
reduction. This reflects both filtering and the change in file format.

The raw ingestion guard skipped the upload when the file already
existed; the file path, size, and modification time stayed the same.
The cleaning rerun reused the published output and report. A separate
missing-column test rejected the bad input and left the published
files unchanged.
### How to run the data preparation notebooks

1. In the course Databricks workspace, attach the notebooks to the
   course lab cluster. Confirm you can access the project storage
   and the `dsai3202_databricks.proj_bookrating` schema.
2. Run [`04_verify_ingestion_rerun`](notebooks/04_verify_ingestion_rerun.py)
   to check that the existing Goodreads v1 CSV is left in place.
   This check assumes the raw CSV has already been uploaded.
3. Run [`01_clean_goodreads`](notebooks/01_clean_goodreads.py) from top
   to bottom. The notebook validates the raw data, keeps explicit
   ratings from 1 to 5, and publishes or reuses the cleaned table.
4. Run [`03_eda_goodreads`](notebooks/03_eda_goodreads.py) after the
   cleaned table is available. It reads the published table and
   does not depend on the feature notebook.

### Exploratory data analysis

The full cleaned dataset contains 104,551,549 explicit ratings
from 816,371 readers across 2,325,541 books. The mean rating was
3.939 stars, and 69.83% of ratings were 4 or 5 stars.

The median reader had 55 ratings, while the median book had 4.
About 51.34% of books and 8.09% of readers had fewer than five
ratings. An average based on a short rating history can be
unstable, so the planned model uses counts alongside averages.

The Pearson correlation between log(1 + book rating count) and
average book rating was 0.0219, indicating little linear
association on that scale. These are descriptive results from
the full dataset. Model features will use training ratings only,

## AI Use

| Contributor | AI tool | What I asked | What it returned | What I changed or verified |
| --- | --- | --- | --- | --- |
| Aysha Ghannam | ChatGPT | Help writing and debugging the PySpark and SQL cleaning code. | Drafted data validation, duplicate checks, rating filters, Parquet export, and table registration code. Suggested a fix for the unsupported write mode. | I applied the fix and ran the notebook in Databricks. I verified that the output contained 104,551,549 ratings and that rerunning reused the existing data and report. |
| Aysha Ghannam | ChatGPT | Help writing a missing-column test and checking the raw ingestion rerun guard. | Drafted the validation test and a check that compares raw file metadata before and after skipping the upload. | I verified that the test rejected the input missing `rating` and left the published files unchanged. The raw CSV’s path, size, and modification time also stayed unchanged. |
| Aysha Ghannam | ChatGPT | Help writing the exploratory data analysis (EDA) code. | Drafted code for rating distributions, reader and book activity, correlation, and storage measurements. | I ran the analysis on the full cleaned dataset and checked the outputs. The 88.65% size reduction reflects both filtering and the change in file format. |


and each training row's own rating will be excluded from its
history features.
