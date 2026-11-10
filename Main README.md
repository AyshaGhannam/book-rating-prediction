# Book Rating Prediction

**Course:** DSAI3202 Fall 2026, Project Phase 1
**Team:** Aysha Ghannam (60308393) and Nour Ben Ayed (60305269)


No credentials are stored in this repository. The dataset is stored in Azure and is not committed to Git.

---

## 1. Project and hypothesis

- **Problem:** predict the 1-5 star rating a reader would give a book.
- **Target:** `rating`, an explicit whole-star rating from 1 to 5.
- **Baseline plan:** predict the book's average training rating. For books unseen in training, predict the overall training average.
- **Hypothesis:** a Random Forest using reader and book rating histories will achieve a lower test mean absolute error (MAE) than the baseline.
- **Approved refinements:** none.
- **Scope:** Phase 1 builds and validates the data pipeline and features. Model training and evaluation are Phase 2.

The dataset is Goodreads (UCSD Book Graph). 

---

## 2. Dataset and architecture

### Source and ingestion

| Item | Detail |
|---|---|
| Dataset | Goodreads interactions, UCSD Book Graph |
| Source link | https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/goodreads_interactions.csv |
| Format and size | CSV, 4,318,621,741 bytes (4.319 GB, shown as 4.02 GiB in Azure) |
| Version | `v1` |
| Raw schema | `user_id`, `book_id`, `is_read`, `rating`, `is_reviewed` |
| Ingestion mode | Batch: download once, upload to the Azure raw container |
| Refresh strategy | Fixed snapshot. A new release would go to a new versioned path (for example `v2`); `v1` is never overwritten. |
| Manifest | [`manifests/manifest_v1.json`](manifests/manifest_v1.json) |
| SHA-256 | `A204BCD79E3DEC594B3D8DA6CBFF68598D121C087FF38E7A05589C019EC830EB` (the original and the copy downloaded back from Azure matched) |

**Repeat ingestion.** [`notebooks/04_verify_ingestion_rerun.py`](notebooks/04_verify_ingestion_rerun.py) finds exactly one raw file, checks its size against the manifest, skips the upload, and asserts that path,
