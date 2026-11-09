# Databricks notebook source
# MAGIC %sql
# MAGIC SELECT user_id, book_id, rating
# MAGIC FROM dsai3202_databricks.proj_bookrating.bookrating_ratings_v1
# MAGIC LIMIT 5;

# COMMAND ----------

# step 2 - split the data into train / validation / test
from pyspark.sql import functions as F

# the cleaned table from Aicha
TABLE = "dsai3202_databricks.proj_bookrating.bookrating_ratings_v1"

# I only need these 3 columns, rating as double so I can average it later
raw = spark.table(TABLE).select("user_id", "book_id", F.col("rating").cast("double").alias("rating"))

# hash of user_id + book_id gives a number 0-99, same pair = same number every run
bucket = F.pmod(F.xxhash64(F.col("user_id").cast("string"), F.col("book_id").cast("string")), F.lit(100))

# 0-79 train, 80-89 validation, 90-99 test
df = (raw.withColumn("bucket", bucket)
         .withColumn("split", F.when(F.col("bucket") < 80, "train")
                               .when(F.col("bucket") < 90, "validation")
                               .otherwise("test"))
         .drop("bucket"))

# count rows in each split and in total
total = raw.count()
counts = {r["split"]: r["count"] for r in df.groupBy("split").count().collect()}
print("total:", total, "| counts:", counts, "| sum:", sum(counts.values()))

# should be around 80 / 10 / 10
for s, c in counts.items():
    print(f"{s}: {c / total:.4%}")

# total has to be 104,551,549 and the 3 splits have to add up to it
assert total == 104_551_549
assert sum(counts.values()) == 104_551_549

# a pair should not be in 2 splits or show up twice, this should be 0
multi = (df.groupBy("user_id", "book_id")
           .agg(F.countDistinct("split").alias("n_splits"), F.count("*").alias("n_rows"))
           .filter("n_splits > 1 OR n_rows > 1")
           .count())
print("pairs in several splits or repeated:", multi)
assert multi == 0

# COMMAND ----------

# step 3 - 4 features, calculated from TRAIN ratings only

# overall train average (used when a reader or book has no other ratings)
train = df.filter(F.col("split") == "train")
tot = train.agg(F.count("*").alias("n"), F.sum("rating").alias("s")).first()
N, S = tot["n"], float(tot["s"])
print("train rows:", N, "| train rating sum:", S)

# sum and count of ratings per reader (train only)
u = train.groupBy("user_id").agg(F.sum("rating").alias("u_sum"), F.count("*").alias("u_cnt"))
# sum and count of ratings per book (train only)
b = train.groupBy("book_id").agg(F.sum("rating").alias("b_sum"), F.count("*").alias("b_cnt"))

# add them to every row, readers/books not in train get 0 (unseen)
j = (df.join(u, "user_id", "left")
       .join(b, "book_id", "left")
       .fillna(0, subset=["u_sum", "u_cnt", "b_sum", "b_cnt"]))

is_train = F.col("split") == "train"
r = F.col("rating")

# for train rows take away the row's own rating and 1 from the count (so the answer doesn't leak)
# for validation/test use the full train numbers
u_sum = F.when(is_train, F.col("u_sum") - r).otherwise(F.col("u_sum"))
u_cnt = F.when(is_train, F.col("u_cnt") - 1).otherwise(F.col("u_cnt"))
b_sum = F.when(is_train, F.col("b_sum") - r).otherwise(F.col("b_sum"))
b_cnt = F.when(is_train, F.col("b_cnt") - 1).otherwise(F.col("b_cnt"))

# overall train average: for train rows leave out the row itself, for others use all of train
g_avg = F.when(is_train, (F.lit(S) - r) / F.lit(float(N - 1))).otherwise(F.lit(S / N))

# if count is 0 use the overall average and keep the count at 0
feat = (j.withColumn("reader_rating_count", u_cnt.cast("long"))
         .withColumn("book_rating_count", b_cnt.cast("long"))
         .withColumn("reader_avg_rating", F.when(u_cnt > 0, u_sum / u_cnt).otherwise(g_avg))
         .withColumn("book_avg_rating", F.when(b_cnt > 0, b_sum / b_cnt).otherwise(g_avg))
         .select("user_id", "book_id", "rating", "split",   # rating stays as the target
                 "reader_avg_rating", "reader_rating_count",
                 "book_avg_rating", "book_rating_count"))

feat.printSchema()
display(feat.limit(10))

# COMMAND ----------

# step 4a - small example: does a train row's own rating get left out?

def make_features(df):
    # same as step 3, just put in a function so I can test it on a tiny table
    train = df.filter(F.col("split") == "train")
    tot = train.agg(F.count("*").alias("n"), F.sum("rating").alias("s")).first()
    N, S = tot["n"], float(tot["s"])
    u = train.groupBy("user_id").agg(F.sum("rating").alias("u_sum"), F.count("*").alias("u_cnt"))
    b = train.groupBy("book_id").agg(F.sum("rating").alias("b_sum"), F.count("*").alias("b_cnt"))
    j = (df.join(u, "user_id", "left").join(b, "book_id", "left")
           .fillna(0, subset=["u_sum", "u_cnt", "b_sum", "b_cnt"]))
    is_train, r = F.col("split") == "train", F.col("rating")
    u_sum = F.when(is_train, F.col("u_sum") - r).otherwise(F.col("u_sum"))
    u_cnt = F.when(is_train, F.col("u_cnt") - 1).otherwise(F.col("u_cnt"))
    b_sum = F.when(is_train, F.col("b_sum") - r).otherwise(F.col("b_sum"))
    b_cnt = F.when(is_train, F.col("b_cnt") - 1).otherwise(F.col("b_cnt"))
    g_avg = F.when(is_train, (F.lit(S) - r) / F.lit(float(N - 1))).otherwise(F.lit(S / N))
    return (j.withColumn("reader_rating_count", u_cnt.cast("long"))
             .withColumn("book_rating_count", b_cnt.cast("long"))
             .withColumn("reader_avg_rating", F.when(u_cnt > 0, u_sum / u_cnt).otherwise(g_avg))
             .withColumn("book_avg_rating", F.when(b_cnt > 0, b_sum / b_cnt).otherwise(g_avg))
             .select("user_id", "book_id", "rating", "split",
                     "reader_avg_rating", "reader_rating_count",
                     "book_avg_rating", "book_rating_count"))

# tiny table, I put the splits by hand so I know the right answers
# train total: N = 3 rows, sum = 5 + 3 + 1 = 9
tiny = spark.createDataFrame([
    ("A", "X", 5.0, "train"), ("A", "Y", 3.0, "train"), ("B", "X", 1.0, "train"),
    ("A", "Z", 4.0, "validation"), ("C", "X", 2.0, "test"),
], ["user_id", "book_id", "rating", "split"])

t = {(x.user_id, x.book_id): x for x in make_features(tiny).collect()}
def feats(k):
    x = t[k]
    return (x.reader_avg_rating, x.reader_rating_count, x.book_avg_rating, x.book_rating_count)

# A-X (rating 5): reader A without it is just A-Y = 3, book X without it is just B-X = 1
assert feats(("A", "X")) == (3.0, 1, 1.0, 1)
# A-Y (rating 3): reader A without it = 5, book Y has nobody else -> (9-3)/2 = 3.0 and count 0
assert feats(("A", "Y")) == (5.0, 1, 3.0, 0)
# B-X (rating 1): reader B has nobody else -> (9-1)/2 = 4.0 and count 0, book X without it = 5
assert feats(("B", "X")) == (4.0, 0, 5.0, 1)
# validation A-Z: full train numbers for A = (5+3)/2 = 4.0, book Z unseen -> 9/3 = 3.0, count 0
assert feats(("A", "Z")) == (4.0, 2, 3.0, 0)
# test C-X: reader C unseen -> 3.0, count 0, book X full train = (5+1)/2 = 3.0
assert feats(("C", "X")) == (3.0, 0, 3.0, 2)
print("small example passed - own rating is left out for train rows")

# use this tested function for the real data
feat = make_features(df)

# COMMAND ----------

# step 4b - write to a staging folder first, check it, publish only if all checks pass
import uuid
SPLITS = ["train", "validation", "test"]
BASE = "abfss://curated@bookrating60308393.dfs.core.windows.net/goodreads/v1/features/"
run_tag = uuid.uuid4().hex[:8]                       # id for this run
stage = "abfss://curated@bookrating60308393.dfs.core.windows.net/goodreads/v1/features_staging/" + run_tag + "/"

# save each split as its own parquet folder (in staging, not the real folder yet)
for s in SPLITS:
    feat.filter(F.col("split") == s).write.mode("overwrite").parquet(stage + s + "/")

# read it back from staging, this is what I check
staged = {s: spark.read.parquet(stage + s + "/") for s in SPLITS}
staged_all = staged["train"].unionByName(staged["validation"]).unionByName(staged["test"])

# no rows lost or added, total and per split
assert staged_all.count() == 104_551_549
for s in SPLITS:
    assert staged[s].count() == counts[s], s
# no pair twice after the joins
assert staged_all.groupBy("user_id", "book_id").count().filter("count > 1").count() == 0

# features: no missing, no NaN / infinite, averages in 1-5, counts >= 0
cols = ["reader_avg_rating", "reader_rating_count", "book_avg_rating", "book_rating_count"]
bad = None
for c in cols:
    cond = F.col(c).isNull() | F.isnan(F.col(c).cast("double")) | F.col(c).cast("double").isin(float("inf"), float("-inf"))
    bad = cond if bad is None else (bad | cond)
bad = (bad | (F.col("reader_avg_rating") < 1) | (F.col("reader_avg_rating") > 5)
           | (F.col("book_avg_rating") < 1) | (F.col("book_avg_rating") > 5)
           | (F.col("reader_rating_count") < 0) | (F.col("book_rating_count") < 0))
n_bad = staged_all.filter(bad).count()
print("rows with a bad feature:", n_bad)
assert n_bad == 0
print("validation passed, staging folder:", stage)

# COMMAND ----------

# step 4c - validation passed, so copy staging to the real folders and register the tables
CATALOG_SCHEMA = "dsai3202_databricks.proj_bookrating"

for s in SPLITS:
    dbutils.fs.rm(BASE + s + "/", True)                      # remove old output so a rerun doesn't double the data
    dbutils.fs.cp(stage + s + "/", BASE + s + "/", True)     # copy the checked files in
dbutils.fs.rm(stage, True)                                   # clean up staging

# register the 3 tables, they point at the parquet folders
for s in SPLITS:
    name = f"{CATALOG_SCHEMA}.features_{s}"
    spark.sql(f"DROP TABLE IF EXISTS {name}")
    spark.sql(f"CREATE TABLE {name} USING PARQUET LOCATION '{BASE}{s}/'")
    print(name, "->", BASE + s + "/", "| rows:", spark.table(name).count())

# COMMAND ----------

# MAGIC %sql
# MAGIC REFRESH TABLE dsai3202_databricks.proj_bookrating.features_train;
# MAGIC REFRESH TABLE dsai3202_databricks.proj_bookrating.features_validation;
# MAGIC REFRESH TABLE dsai3202_databricks.proj_bookrating.features_test;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 'train' AS split, COUNT(*) AS n FROM dsai3202_databricks.proj_bookrating.features_train
# MAGIC UNION ALL SELECT 'validation', COUNT(*) FROM dsai3202_databricks.proj_bookrating.features_validation
# MAGIC UNION ALL SELECT 'test', COUNT(*) FROM dsai3202_databricks.proj_bookrating.features_test;