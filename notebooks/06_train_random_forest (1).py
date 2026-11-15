# Databricks notebook source
import os, platform
import numpy as np
import pandas as pd
import sklearn
import mlflow

print("python", platform.python_version())
print("spark", spark.version)
print("numpy", np.__version__)
print("pandas", pd.__version__)
print("scikit-learn", sklearn.__version__)
print("mlflow", mlflow.__version__)
print("driver CPU cores:", os.cpu_count())

# COMMAND ----------

import json, time
import numpy as np
import pandas as pd
import mlflow
from pyspark.sql import functions as F

ROOT = "abfss://curated@bookrating60308393.dfs.core.windows.net/goodreads/v1/"
FROZEN = ROOT + "features_frozen_v1/"

FEATURES = ["reader_avg_rating", "reader_rating_count",
            "book_avg_rating", "book_rating_count"]
TARGET = "rating"
SEED = 42

# Sampling rule: hash of (user_id, book_id), keep rows whose bucket (0-999) is below the cutoff.
# The same rows are picked on every run because the frozen data never changes.
TRAIN_CUTOFF = 25    # about 2.5% of training rows
VAL_CUTOFF = 100     # about 10% of validation rows

def bucket():
    return F.pmod(
        F.xxhash64(F.col("user_id").cast("string"),
                   F.col("book_id").cast("string"),
                   F.lit("sample-v1")),
        F.lit(1000))

cols = ["user_id", "book_id"] + FEATURES + [TARGET]

train_pd = (spark.read.parquet(FROZEN + "train/")
            .filter(bucket() < TRAIN_CUTOFF).select(*cols).toPandas())
val_pd = (spark.read.parquet(FROZEN + "validation/")
          .filter(bucket() < VAL_CUTOFF).select(*cols).toPandas())

print("train sample rows:     ", f"{len(train_pd):,}")
print("validation sample rows:", f"{len(val_pd):,}")
print()
print(train_pd[FEATURES + [TARGET]].describe().T[["mean", "min", "max"]])

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from mlflow.models import infer_signature
import sklearn

# float64 everywhere, so the saved model's input schema is the same for every caller
X_train = train_pd[FEATURES].astype("float64")
y_train = train_pd[TARGET].astype("float64")
X_val = val_pd[FEATURES].astype("float64")
y_val = val_pd[TARGET].astype("float64")

# baselines on this same validation sample
base_book = mean_absolute_error(y_val, X_val["book_avg_rating"])
base_reader = mean_absolute_error(y_val, X_val["reader_avg_rating"])
print(f"Validation sample baseline MAE: book average {base_book:.4f} | reader average {base_reader:.4f}")

def train_and_log(params, run_name):
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({**params, "seed": SEED})
        mlflow.set_tags({
            "data_version": "features_frozen_v1",
            "train_sample_rows": len(train_pd),
            "val_sample_rows": len(val_pd),
            "sample_rule": f"xxhash64(user_id, book_id, 'sample-v1') % 1000 < cutoff (train {TRAIN_CUTOFF}, val {VAL_CUTOFF})",
            "features": ",".join(FEATURES),
            "sklearn": sklearn.__version__,
            "mlflow": mlflow.__version__,
        })
        t0 = time.time()
        model = RandomForestRegressor(**params, random_state=SEED, n_jobs=4)
        model.fit(X_train, y_train)
        fit_s = time.time() - t0

        pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, pred)
        rmse = mean_squared_error(y_val, pred) ** 0.5
        mlflow.log_metrics({
            "val_sample_mae": mae,
            "val_sample_rmse": rmse,
            "val_sample_baseline_book_mae": base_book,
            "fit_seconds": fit_s,
        })
        sig = infer_signature(X_val.head(5), model.predict(X_val.head(5)))
        mlflow.sklearn.log_model(model, name="model", signature=sig)
        print(f"{run_name}: MAE {mae:.4f} | RMSE {rmse:.4f} | fit {fit_s:.0f}s | run_id {run.info.run_id}")
        return run.info.run_id, mae

params_v1 = dict(n_estimators=100, max_depth=12, min_samples_leaf=50, max_features=1.0)
run_id_v1, mae_v1 = train_and_log(params_v1, "rf_v1_first_model")

# COMMAND ----------

tuning_grid = [
    {"max_depth": 12, "min_samples_leaf": 50},
    {"max_depth": 16, "min_samples_leaf": 50},
    {"max_depth": 12, "min_samples_leaf": 200},
    {"max_depth": 20, "min_samples_leaf": 100},
]

tuning_results = []
for i, g in enumerate(tuning_grid, start=1):
    params = dict(n_estimators=50, max_features=1.0, **g)
    name = f"rf_tune_{i}_depth{g['max_depth']}_leaf{g['min_samples_leaf']}"
    run_id, mae = train_and_log(params, name)
    tuning_results.append({"run": i, **g, "n_estimators": 50,
                           "val_sample_mae": round(mae, 4), "run_id": run_id})

print()
print(pd.DataFrame(tuning_results).sort_values("val_sample_mae").to_string(index=False))

# COMMAND ----------

import json, time
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from pyspark.sql import functions as F

ROOT = "abfss://curated@bookrating60308393.dfs.core.windows.net/goodreads/v1/"
FROZEN = ROOT + "features_frozen_v1/"

FEATURES = ["reader_avg_rating", "reader_rating_count",
            "book_avg_rating", "book_rating_count"]
TARGET = "rating"
SEED = 42

print("ready")

# COMMAND ----------

import json, time
import numpy as np
import mlflow
import mlflow.sklearn
from pyspark.sql import functions as F

ROOT = "abfss://curated@bookrating60308393.dfs.core.windows.net/goodreads/v1/"
FROZEN = ROOT + "features_frozen_v1/"
FEATURES = ["reader_avg_rating", "reader_rating_count",
            "book_avg_rating", "book_rating_count"]
TARGET = "rating"
FINAL_RUN_ID = "b80c285cc4ea42febd3ef8e5cc88e1a0"
LOGGED_MODEL_ID = "m-acf95ca40bf34137878254f901870a15"
N_CHUNKS = 10
EXPECTED_TEST = 10_455_821
PRED_PATH = ROOT + "model_results/test_predictions_v1/"

# load the final model: try the run first, then the logged model's own address
try:
    model = mlflow.sklearn.load_model(f"runs:/{FINAL_RUN_ID}/model")
    print("model loaded from run")
except Exception as e1:
    print("loading from run failed:", str(e1)[:200])
    lm = mlflow.get_logged_model(LOGGED_MODEL_ID)
    model = mlflow.sklearn.load_model(lm.model_uri)
    print("model loaded from", lm.model_uri)

test_sdf = spark.read.parquet(FROZEN + "test/").select("user_id", "book_id", *FEATURES, TARGET)
chunk_id = F.pmod(
    F.xxhash64(F.col("user_id").cast("string"), F.col("book_id").cast("string"), F.lit("eval-chunk-v1")),
    F.lit(N_CHUNKS))
train_mean = spark.read.parquet(FROZEN + "train/").agg(F.avg("rating")).first()[0]

# running totals, so only small numbers stay in memory
acc = {k: 0.0 for k in ["n", "rf_abs", "rf_sq", "book_abs", "book_sq",
                        "reader_abs", "reader_sq", "glob_abs", "glob_sq", "d_sum", "d_sq"]}

t0 = time.time()
for k in range(N_CHUNKS):
    pdf = test_sdf.filter(chunk_id == k).toPandas()
    y = pdf[TARGET].to_numpy(dtype="float64")
    p_rf = model.predict(pdf[FEATURES].astype("float64"))
    p_book = pdf["book_avg_rating"].to_numpy(dtype="float64")
    p_reader = pdf["reader_avg_rating"].to_numpy(dtype="float64")

    e_rf, e_book = y - p_rf, y - p_book
    e_reader, e_glob = y - p_reader, y - train_mean
    d = np.abs(e_book) - np.abs(e_rf)   # positive: Random Forest closer than the baseline

    acc["n"] += len(y)
    acc["rf_abs"] += float(np.abs(e_rf).sum());       acc["rf_sq"] += float((e_rf ** 2).sum())
    acc["book_abs"] += float(np.abs(e_book).sum());   acc["book_sq"] += float((e_book ** 2).sum())
    acc["reader_abs"] += float(np.abs(e_reader).sum()); acc["reader_sq"] += float((e_reader ** 2).sum())
    acc["glob_abs"] += float(np.abs(e_glob).sum());   acc["glob_sq"] += float((e_glob ** 2).sum())
    acc["d_sum"] += float(d.sum());                   acc["d_sq"] += float((d ** 2).sum())

    out = pdf[["user_id", "book_id"] + FEATURES + [TARGET]].copy()
    out["pred_rf"] = p_rf
    out["pred_book_baseline"] = p_book
    spark.createDataFrame(out).write.mode("overwrite" if k == 0 else "append").parquet(PRED_PATH)

    print(f"chunk {k + 1}/{N_CHUNKS}: {len(y):,} rows | total so far {int(acc['n']):,} | {time.time() - t0:.0f}s")
    del pdf, y, p_rf, p_book, p_reader, e_rf, e_book, e_reader, e_glob, d, out

n = int(acc["n"])
assert n == EXPECTED_TEST, f"expected {EXPECTED_TEST:,} rows, got {n:,}"

mean_d = acc["d_sum"] / n
var_d = (acc["d_sq"] - n * mean_d ** 2) / (n - 1)
se_d = (var_d / n) ** 0.5

def m(prefix):
    return {"mae": round(acc[prefix + "_abs"] / n, 4),
            "rmse": round((acc[prefix + "_sq"] / n) ** 0.5, 4)}

results = {
    "split": "test",
    "rows": n,
    "final_run_id": FINAL_RUN_ID,
    "logged_model_id": LOGGED_MODEL_ID,
    "data_version": "features_frozen_v1",
    "random_forest": m("rf"),
    "baseline_book_average": m("book"),
    "reference_reader_average": m("reader"),
    "reference_global_mean": m("glob"),
    "mae_improvement_vs_book_baseline": {
        "mean": round(mean_d, 4),
        "ci95_low": round(mean_d - 1.96 * se_d, 4),
        "ci95_high": round(mean_d + 1.96 * se_d, 4),
        "relative_percent": round(100 * mean_d / (acc["book_abs"] / n), 2),
    },
}
print(json.dumps(results, indent=2))

dbutils.fs.put(ROOT + "model_results/test_results_v1.json", json.dumps(results, indent=2), True)
print("saved results file and predictions")

# COMMAND ----------

from pyspark.sql import functions as F

ROOT = "abfss://curated@bookrating60308393.dfs.core.windows.net/goodreads/v1/"
preds = spark.read.parquet(ROOT + "model_results/test_predictions_v1/")

n = preds.count()
print("prediction rows:", f"{n:,}")
assert n == 10_455_821

preds = (preds
    .withColumn("err_rf", F.abs(F.col("rating") - F.col("pred_rf")))
    .withColumn("err_book", F.abs(F.col("rating") - F.col("pred_book_baseline")))
    .withColumn("book_history",
        F.when(F.col("book_rating_count") == 0, "a) 0 (unseen book)")
         .when(F.col("book_rating_count") < 5, "b) 1-4")
         .when(F.col("book_rating_count") < 20, "c) 5-19")
         .when(F.col("book_rating_count") < 100, "d) 20-99")
         .when(F.col("book_rating_count") < 1000, "e) 100-999")
         .otherwise("f) 1000+")))
preds.cache()

# error slice: how many training ratings the book has
slice_book = (preds.groupBy("book_history")
    .agg(F.count("*").alias("rows"),
         F.round(F.avg("err_book"), 4).alias("baseline_mae"),
         F.round(F.avg("err_rf"), 4).alias("rf_mae"))
    .withColumn("share_of_test_%", F.round(100 * F.col("rows") / n, 2))
    .withColumn("improvement", F.round(F.col("baseline_mae") - F.col("rf_mae"), 4))
    .orderBy("book_history"))
display(slice_book)

# COMMAND ----------

by_rating = (preds.groupBy("rating")
    .agg(F.count("*").alias("rows"),
         F.round(F.avg("err_book"), 4).alias("baseline_mae"),
         F.round(F.avg("err_rf"), 4).alias("rf_mae"),
         F.round(F.avg("pred_rf"), 3).alias("avg_prediction"))
    .withColumn("share_of_test_%", F.round(100 * F.col("rows") / n, 2))
    .orderBy("rating"))
display(by_rating)

# COMMAND ----------

cols = ["user_id", "book_id", "rating", "pred_rf", "pred_book_baseline",
        "reader_avg_rating", "reader_rating_count",
        "book_avg_rating", "book_rating_count", "err_rf"]

print("10 largest Random Forest errors")
display(preds.orderBy(F.desc("err_rf")).select(*cols).limit(10))

print("10 random test rows (seed 42)")
display(preds.sample(False, 30 / n, 42).select(*cols).limit(10))

# COMMAND ----------

import json, os, tempfile
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from pyspark.sql import functions as F

ROOT = "abfss://curated@bookrating60308393.dfs.core.windows.net/goodreads/v1/"
FEATURES = ["reader_avg_rating", "reader_rating_count",
            "book_avg_rating", "book_rating_count"]
FINAL_RUN_ID = "b80c285cc4ea42febd3ef8e5cc88e1a0"

# download the final model's folder to a temporary folder (read-only use)
local_path = mlflow.artifacts.download_artifacts(
    artifact_uri=f"runs:/{FINAL_RUN_ID}/model", dst_path=tempfile.mkdtemp())
for f in sorted(os.listdir(local_path)):
    p = os.path.join(local_path, f)
    if os.path.isfile(p):
        print(f"{f}: {os.path.getsize(p):,} bytes")

print("\nrequirements.txt:\n" + open(os.path.join(local_path, "requirements.txt")).read())
print("MLmodel:\n" + open(os.path.join(local_path, "MLmodel")).read())

# fixed example inputs, chosen by a hash rule so the same rows come back every time
model = mlflow.sklearn.load_model(local_path)
preds = spark.read.parquet(ROOT + "model_results/test_predictions_v1/")
h = F.xxhash64(F.col("user_id").cast("string"), F.col("book_id").cast("string"), F.lit("handoff-v1"))
pool = preds.withColumn("h", h)

def pick(cond, k):
    return pool.filter(cond).orderBy("h").limit(k)

sel = (pick("book_rating_count = 0", 3)                                      # unseen books
       .unionByName(pick("reader_rating_count = 0", 3))                      # unseen readers
       .unionByName(pick("book_rating_count BETWEEN 1 AND 4", 4))            # short book history
       .unionByName(pick("book_rating_count >= 100 AND reader_rating_count >= 20", 10))  # typical
       .dropDuplicates(["user_id", "book_id"]).orderBy("h"))
fx = sel.toPandas().reset_index(drop=True)
fx.insert(0, "row_id", [f"r{i + 1:02d}" for i in range(len(fx))])

expected = model.predict(fx[FEATURES].astype("float64"))

# the downloaded model must reproduce the predictions saved during the test evaluation
max_diff = float(np.abs(expected - fx["pred_rf"].to_numpy()).max())
print(f"rows: {len(fx)} | max difference from the saved test predictions: {max_diff:.2e}")
assert max_diff < 1e-9

fixed = fx[["row_id", "user_id", "book_id"] + FEATURES].copy()
fixed["expected_prediction"] = expected
csv_text = fixed.to_csv(index=False)
dbutils.fs.put(ROOT + "handoff/fixed_inputs_v1.csv", csv_text, True)
print("saved to storage: handoff/fixed_inputs_v1.csv\n")
print(csv_text)

# COMMAND ----------

import json
from datetime import datetime, timezone
from pyspark.sql import functions as F

ROOT = "abfss://curated@bookrating60308393.dfs.core.windows.net/goodreads/v1/"
FROZEN = ROOT + "features_frozen_v1/"
AGG = ROOT + "feature_lookup_v1/"
SCHEMA = "dsai3202_databricks.proj_bookrating"

def exists(path):
    try:
        dbutils.fs.ls(path)
        return True
    except Exception:
        return False

if exists(AGG + "_manifest.json"):
    print("LOOKUP ALREADY SAVED: nothing written.")
else:
    train = spark.read.parquet(FROZEN + "train/").select("user_id", "book_id", "rating")
    tot = train.agg(F.count("*").alias("n"), F.sum("rating").alias("s")).first()
    N, S = int(tot["n"]), float(tot["s"])

    (train.groupBy("user_id")
          .agg(F.sum("rating").cast("double").alias("reader_sum"), F.count("*").alias("reader_cnt"))
          .write.mode("overwrite").parquet(AGG + "reader/"))
    (train.groupBy("book_id")
          .agg(F.sum("rating").cast("double").alias("book_sum"), F.count("*").alias("book_cnt"))
          .write.mode("overwrite").parquet(AGG + "book/"))

    manifest = {
        "name": "feature_lookup_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": FROZEN + "train/",
        "train_rows": N,
        "train_rating_sum": S,
        "global_mean_S_over_N": S / N,
        "readers": spark.read.parquet(AGG + "reader/").count(),
        "books": spark.read.parquet(AGG + "book/").count(),
    }
    dbutils.fs.put(AGG + "_manifest.json", json.dumps(manifest, indent=2), False)
    print("LOOKUP SAVED")

manifest = json.loads(dbutils.fs.head(AGG + "_manifest.json", 65536))
print(json.dumps(manifest, indent=2))
G = manifest["global_mean_S_over_N"]

for name, folder in [("reader_train_agg_v1", "reader/"), ("book_train_agg_v1", "book/")]:
    spark.sql(f"CREATE TABLE IF NOT EXISTS {SCHEMA}.{name} USING PARQUET LOCATION '{AGG}{folder}'")

# check: recompute the features of the 20 fixed inputs from the lookup and compare
fx = (spark.read.option("header", True).option("inferSchema", True)
      .csv(ROOT + "handoff/fixed_inputs_v1.csv"))
rd = spark.read.parquet(AGG + "reader/")
bk = spark.read.parquet(AGG + "book/")

chk = (fx.join(rd, "user_id", "left").join(bk, "book_id", "left")
       .withColumn("r_cnt", F.coalesce(F.col("reader_cnt"), F.lit(0)))
       .withColumn("b_cnt", F.coalesce(F.col("book_cnt"), F.lit(0)))
       .withColumn("r_avg", F.when(F.col("r_cnt") > 0, F.col("reader_sum") / F.col("r_cnt")).otherwise(F.lit(G)))
       .withColumn("b_avg", F.when(F.col("b_cnt") > 0, F.col("book_sum") / F.col("b_cnt")).otherwise(F.lit(G))))

diffs = chk.agg(
    F.count("*").alias("rows"),
    F.max(F.abs(F.col("r_avg") - F.col("reader_avg_rating").cast("double"))).alias("max_reader_avg_diff"),
    F.max(F.abs(F.col("b_avg") - F.col("book_avg_rating").cast("double"))).alias("max_book_avg_diff"),
    F.max(F.abs(F.col("r_cnt") - F.col("reader_rating_count").cast("double"))).alias("max_reader_cnt_diff"),
    F.max(F.abs(F.col("b_cnt") - F.col("book_rating_count").cast("double"))).alias("max_book_cnt_diff"),
).first().asDict()
print(diffs)
assert diffs["rows"] == 20
assert all(v < 1e-9 for k, v in diffs.items() if k != "rows")
print("PASS: the lookup reproduces the features of all 20 fixed inputs")