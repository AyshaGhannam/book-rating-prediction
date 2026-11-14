# Databricks notebook source
import json
from datetime import datetime, timezone
from pyspark.sql import functions as F

SPLITS = ["train", "validation", "test"]
ROOT = "abfss://curated@bookrating60308393.dfs.core.windows.net/goodreads/v1/"
LIVE = ROOT + "features/"
FROZEN = ROOT + "features_frozen_v1/"
SCHEMA = "dsai3202_databricks.proj_bookrating"
EXPECTED = {"train": 83_638_178, "validation": 10_457_550, "test": 10_455_821}

def exists(path):
    try:
        dbutils.fs.ls(path)
        return True
    except Exception:
        return False

print("LIVE:  ", LIVE)
print("FROZEN:", FROZEN)
print("Live folders found:", [exists(LIVE + s + "/") for s in SPLITS])

# COMMAND ----------

manifest_path = FROZEN + "_manifest.json"

if exists(manifest_path):
    print("ALREADY FROZEN: nothing copied.")
else:
    # check the live folders still match the Phase 1 counts before freezing
    for s in SPLITS:
        n = spark.read.parquet(LIVE + s + "/").count()
        assert n == EXPECTED[s], f"{s}: {n:,} rows, expected {EXPECTED[s]:,}"
        print(f"{s}: {n:,} rows OK")

    # no manifest means any existing frozen folder is a partial copy, so start clean
    if exists(FROZEN):
        dbutils.fs.rm(FROZEN, True)

    for s in SPLITS:
        dbutils.fs.cp(LIVE + s + "/", FROZEN + s + "/", True)
        print(f"copied {s}")

    manifest = {
        "name": "features_frozen_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": LIVE,
        "source_tables": [f"{SCHEMA}.features_{s}" for s in SPLITS],
        "splits": {},
    }
    for s in SPLITS:
        files = [f for f in dbutils.fs.ls(FROZEN + s + "/") if f.name.endswith(".parquet")]
        manifest["splits"][s] = {
            "rows": spark.read.parquet(FROZEN + s + "/").count(),
            "parquet_files": len(files),
            "bytes": sum(f.size for f in files),
        }
        assert manifest["splits"][s]["rows"] == EXPECTED[s], s

    dbutils.fs.put(manifest_path, json.dumps(manifest, indent=2), False)
    print("FROZEN: copy verified and manifest written.")

print(dbutils.fs.head(manifest_path, 65536))

# COMMAND ----------

for s in SPLITS:
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {SCHEMA}.features_{s}_v1_frozen
        USING PARQUET LOCATION '{FROZEN}{s}/'
    """)
    n = spark.table(f"{SCHEMA}.features_{s}_v1_frozen").count()
    assert n == EXPECTED[s], s
    print(f"{SCHEMA}.features_{s}_v1_frozen: {n:,} rows")

# COMMAND ----------

train = spark.read.parquet(FROZEN + "train/")
val = spark.read.parquet(FROZEN + "validation/")

train_mean = train.agg(F.avg("rating")).first()[0]

def metrics(df, pred):
    err = F.col("rating") - pred
    r = df.agg(
        F.count("*").alias("rows"),
        F.avg(F.abs(err)).alias("mae"),
        F.sqrt(F.avg(err * err)).alias("rmse"),
    ).first()
    return {"rows": r["rows"], "mae": round(r["mae"], 4), "rmse": round(r["rmse"], 4)}

results = {
    "split": "validation",
    "train_mean_rating": round(train_mean, 4),
    "baseline_book_average": metrics(val, F.col("book_avg_rating")),
    "reference_reader_average": metrics(val, F.col("reader_avg_rating")),
    "reference_global_mean": metrics(val, F.lit(train_mean)),
}
print(json.dumps(results, indent=2))

dbutils.fs.put(ROOT + "model_results/baseline_validation.json", json.dumps(results, indent=2), True)