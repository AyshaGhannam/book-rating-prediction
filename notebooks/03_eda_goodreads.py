# Databricks notebook source
from pyspark.sql import functions as F

table_name = (
    "dsai3202_databricks.proj_bookrating.bookrating_ratings_v1"
)

ratings = spark.table(table_name)

# Full-dataset rating distribution
distribution = (
    ratings.groupBy("rating")
    .count()
    .orderBy("rating")
    .collect()
)

total_ratings = sum(row["count"] for row in distribution)

summary = [
    (
        row["rating"],
        row["count"],
        round(100 * row["count"] / total_ratings, 2)
    )
    for row in distribution
]

distribution_df = spark.createDataFrame(
    summary,
    ["rating", "number_of_ratings", "percentage"]
)

display(distribution_df)

mean_rating = sum(
    row["rating"] * row["count"]
    for row in distribution
) / total_ratings

print("Scope: FULL cleaned dataset, not a sample")
print(f"Total ratings: {total_ratings:,}")
print(f"Average rating: {mean_rating:.3f}")

# COMMAND ----------

# Full-dataset counts and average ratings per reader and book
reader_stats = ratings.groupBy("user_id").agg(
    F.count("*").alias("rating_count"),
    F.avg("rating").alias("average_rating")
)

book_stats = ratings.groupBy("book_id").agg(
    F.count("*").alias("rating_count"),
    F.avg("rating").alias("average_rating")
)

def summarize_history(dataframe):
    return dataframe.agg(
        F.count("*").alias("total_entities"),
        F.min("rating_count").alias("minimum_ratings"),
        F.round(
            F.avg("rating_count"), 2
        ).alias("mean_ratings"),
        F.expr(
            "percentile_approx(rating_count, 0.5, 10000)"
        ).alias("approx_median_ratings"),
        F.expr(
            "percentile_approx(rating_count, 0.99, 10000)"
        ).alias("approx_99th_percentile"),
        F.max("rating_count").alias("maximum_ratings"),
        F.sum(
            F.when(F.col("rating_count") == 1, 1).otherwise(0)
        ).alias("entities_with_one_rating"),
        F.sum(
            F.when(F.col("rating_count") < 5, 1).otherwise(0)
        ).alias("entities_with_fewer_than_5_ratings")
    )

reader_summary = summarize_history(reader_stats)
book_summary = summarize_history(book_stats)

print("READERS — full cleaned dataset")
display(reader_summary)

print("BOOKS — full cleaned dataset")
display(book_summary)

# COMMAND ----------

def history_buckets(stats, entity):
    return (
        stats.withColumn(
            "bucket_order",
            F.when(F.col("rating_count") == 1, 1)
             .when(F.col("rating_count") < 5, 2)
             .when(F.col("rating_count") < 20, 3)
             .when(F.col("rating_count") < 100, 4)
             .otherwise(5)
        )
        .groupBy("bucket_order")
        .agg(F.count("*").alias("entity_count"))
        .withColumn("entity", F.lit(entity))
    )

history_distribution = (
    history_buckets(reader_stats, "Readers")
    .unionByName(history_buckets(book_stats, "Books"))
    .withColumn(
        "ratings_per_entity",
        F.when(F.col("bucket_order") == 1, "1")
         .when(F.col("bucket_order") == 2, "2–4")
         .when(F.col("bucket_order") == 3, "5–19")
         .when(F.col("bucket_order") == 4, "20–99")
         .otherwise("100+")
    )
    .orderBy("bucket_order", "entity")
    .select("ratings_per_entity", "entity", "entity_count")
)

display(history_distribution)

# COMMAND ----------

# One observation per book, using the full cleaned dataset.
# Log-transform the count so a few heavily rated books
# have less influence on the scale.
book_relationship = (
    book_stats
    .withColumn("log_rating_count", F.log1p(F.col("rating_count")))
    .agg(
        F.count("*").alias("books_analyzed"),
        F.round(
            F.corr("log_rating_count", "average_rating"), 4
        ).alias("pearson_correlation")
    )
)

display(book_relationship)

# COMMAND ----------

raw_dir = (
    "abfss://raw@bookrating60308393.dfs.core.windows.net/goodreads/v1/"
)
parquet_dir = (
    "abfss://processed@bookrating60308393.dfs.core.windows.net/"
    "goodreads/v1/_staging/ratings_a539f0c69fe247f690d2860544c8a9c0/"
)

csv_bytes = next(
    file.size for file in dbutils.fs.ls(raw_dir)
    if file.name == "goodreads_interactions.csv"
)

parquet_files = [
    file for file in dbutils.fs.ls(parquet_dir)
    if file.name.endswith(".parquet")
]
assert parquet_files, "No Parquet data files found"

parquet_bytes = sum(file.size for file in parquet_files)
reduction = 100 * (1 - parquet_bytes / csv_bytes)

print(f"Raw CSV:       {csv_bytes:,} bytes ({csv_bytes / 1e9:.3f} GB)")
print(f"Clean Parquet: {parquet_bytes:,} bytes ({parquet_bytes / 1e9:.3f} GB)")
print(f"Parquet data files: {len(parquet_files)}")
print(f"Raw-to-clean size reduction: {reduction:.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## EDA findings
# MAGIC
# MAGIC The cleaned dataset contains 104,551,549 explicit ratings. The average
# MAGIC rating is 3.939 stars, and 69.83% of ratings are 4 or 5 stars.
# MAGIC
# MAGIC The median reader has 55 ratings, while the median book has 4. About
# MAGIC 51.34% of books have fewer than five ratings, compared with 8.09% of
# MAGIC readers. An average based on only a few ratings can be unstable, so the
# MAGIC model will use rating counts alongside averages.
# MAGIC
# MAGIC Across 2,325,541 books, the Pearson correlation between log(1 + rating
# MAGIC count) and average rating was 0.0219, showing little linear association.
# MAGIC These EDA results use the full cleaned dataset. Model features will use
# MAGIC training data only.
# MAGIC
# MAGIC The raw CSV was 4.319 GB. The cleaned output was 0.490 GB across 33
# MAGIC Snappy-compressed Parquet files, an 88.65% raw-to-clean size reduction.
# MAGIC This compares different row sets and file formats, so it does not
# MAGIC measure compression alone.