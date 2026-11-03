# book-rating-prediction
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
