# Databricks notebook source
# Read the original CSV without changing it
raw_path = (
    "abfss://raw@bookrating60308393.dfs.core.windows.net/"
    "goodreads/v1/goodreads_interactions.csv"
)

raw_df = (
    spark.read
    .option("header", True)
    .csv(raw_path)
)

# Stop if the source columns are unexpected
expected_columns = [
    "user_id", "book_id", "is_read", "rating", "is_reviewed"
]

def validate_columns(dataframe):
    if dataframe.columns != expected_columns:
        raise ValueError(
            f"Unexpected columns: {dataframe.columns}"
        )

validate_columns(raw_df)

# Count every row, grouped by its original rating value
rating_counts = (
    raw_df.groupBy("rating")
    .count()
    .orderBy("rating")
    .collect()
)

print("FULL DATASET — rating counts")
for row in rating_counts:
    print(f"Rating {row['rating']!r}: {row['count']:,} rows")

input_count = sum(row["count"] for row in rating_counts)
print(f"\nTotal input rows: {input_count:,}")

# COMMAND ----------

from pyspark.sql import functions as F

# Safely convert strings to numbers.
# Values that cannot be converted become null.
typed_df = raw_df.select(
    F.expr("try_cast(user_id AS BIGINT)").alias("user_id"),
    F.expr("try_cast(book_id AS BIGINT)").alias("book_id"),
    F.expr("try_cast(is_read AS INT)").alias("is_read"),
    F.expr("try_cast(rating AS INT)").alias("rating"),
    F.expr("try_cast(is_reviewed AS INT)").alias("is_reviewed")
)

# Check each field across the full dataset
checks = typed_df.agg(
    F.count("*").alias("total_rows"),

    F.sum(F.when(
        F.col("user_id").isNull() | (F.col("user_id") < 0), 1
    ).otherwise(0)).alias("invalid_user_ids"),

    F.sum(F.when(
        F.col("book_id").isNull() | (F.col("book_id") < 0), 1
    ).otherwise(0)).alias("invalid_book_ids"),

    F.sum(F.when(
        F.col("rating").isNull() |
        ~F.col("rating").between(0, 5), 1
    ).otherwise(0)).alias("invalid_ratings"),

    F.sum(F.when(
        F.col("is_read").isNull() |
        ~F.col("is_read").isin(0, 1), 1
    ).otherwise(0)).alias("invalid_is_read"),

    F.sum(F.when(
        F.col("is_reviewed").isNull() |
        ~F.col("is_reviewed").isin(0, 1), 1
    ).otherwise(0)).alias("invalid_is_reviewed")
)

typed_df.printSchema()
display(checks)

# COMMAND ----------

# Group by reader and book to find repeated pairs
pair_counts = typed_df.groupBy("user_id", "book_id").count()

duplicate_summary = pair_counts.agg(
    F.count("*").alias("unique_user_book_pairs"),

    F.sum(
        F.when(F.col("count") > 1, 1).otherwise(0)
    ).alias("repeated_pairs"),

    F.sum(F.col("count") - 1).alias("extra_rows")
)

display(duplicate_summary)

# COMMAND ----------

# Separate explicit ratings from unrated interactions
ratings_df = typed_df.filter(F.col("rating").between(1, 5))
unrated_df = typed_df.filter(F.col("rating") == 0)

# Reuse the full-data rating counts from cell 1
unrated_count = sum(
    row["count"] for row in rating_counts
    if row["rating"] == "0"
)

explicit_count = sum(
    row["count"] for row in rating_counts
    if row["rating"] in ["1", "2", "3", "4", "5"]
)

# Quality gates: stop if validity or uniqueness checks fail
validity = checks.first().asDict()
duplicates = duplicate_summary.first().asDict()

assert all(
    value == 0
    for name, value in validity.items()
    if name.startswith("invalid_")
), "FAILED: invalid field values found"

assert duplicates["extra_rows"] == 0, (
    "FAILED: duplicate user–book pairs found"
)

assert explicit_count > 0, "FAILED: no explicit ratings"

assert input_count == explicit_count + unrated_count, (
    "FAILED: row counts do not reconcile"
)

print(f"Input rows:       {input_count:,}")
print(f"Explicit ratings: {explicit_count:,}")
print(f"Unrated rows:     {unrated_count:,}")
print("PASS: validity, uniqueness and row accounting")

# COMMAND ----------

from uuid import uuid4

table_name = (
    "dsai3202_databricks.proj_bookrating.bookrating_ratings_v1"
)

if spark.catalog.tableExists(table_name):
    # Reuse the published output instead of writing another copy
    details = spark.sql(
        f"DESCRIBE TABLE EXTENDED {table_name}"
    ).collect()

    staging_path = next(
        row["data_type"]
        for row in details
        if row["col_name"].strip() == "Location"
    )

    run_id = staging_path.rstrip("/").split("ratings_")[-1]

    print("RERUN: existing table found; no data written.")

else:
    run_id = uuid4().hex

    staging_path = (
        "abfss://processed@bookrating60308393.dfs.core.windows.net/"
        f"goodreads/v1/_staging/ratings_{run_id}/"
    )

    (
        ratings_df.write
        .mode("append")
        .option("compression", "snappy")
        .parquet(staging_path)
    )

    print("FIRST RUN: new Parquet output written.")

# Validate either the existing output or the newly written output
saved_df = spark.read.parquet(staging_path)
saved_count = saved_df.count()

expected_types = [
    (field.name, field.dataType.simpleString())
    for field in ratings_df.schema.fields
]

saved_types = [
    (field.name, field.dataType.simpleString())
    for field in saved_df.schema.fields
]

assert saved_count == explicit_count, "FAILED: row count differs"
assert saved_types == expected_types, "FAILED: column types differ"

print(f"Run ID: {run_id}")
print(f"Saved rows: {saved_count:,}")
print(f"Output path: {staging_path}")
print("PASS: saved row count and column types verified")

# COMMAND ----------

table_name = "dsai3202_databricks.proj_bookrating.bookrating_ratings_v1"

# Register the validated Parquet files as an external table
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {table_name}
    USING PARQUET
    LOCATION '{staging_path}'
""")

# Confirm the table points to this validated output
table_details = spark.sql(
    f"DESCRIBE TABLE EXTENDED {table_name}"
).collect()

table_location = next(
    row["data_type"]
    for row in table_details
    if row["col_name"].strip() == "Location"
)

assert table_location.rstrip("/") == staging_path.rstrip("/"), (
    "The table already points to a different output. "
    "Stop here and check before changing it."
)

display(spark.table(table_name).limit(5))

print(f"PASS: registered {table_name}")
print(f"Data location: {table_location}")

# COMMAND ----------

import json
from datetime import datetime, timezone

report = {
    "run_id": run_id,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source_path": raw_path,
    "input_rows": input_count,
    "unrated_rows": unrated_count,
    "explicit_rating_rows": explicit_count,
    "saved_rows": saved_count,
    "field_checks": validity,
    "duplicate_checks": duplicates,
    "output_path": table_location,
    "table_name": table_name,
    "row_accounting_passed": (
        input_count == explicit_count + unrated_count
    ),
    "saved_count_passed": saved_count == explicit_count,
    "saved_column_types_passed": saved_types == expected_types
}

report_path = (
    "abfss://processed@bookrating60308393.dfs.core.windows.net/"
    f"goodreads/v1/run_reports/cleaning_{run_id}.json"
)

# List existing reports without changing them
report_folder = report_path.rsplit("/", 1)[0] + "/"
existing_paths = {
    item.path for item in dbutils.fs.ls(report_folder)
}

if report_path in existing_paths:
    existing_report = json.loads(
        dbutils.fs.head(report_path, 65536)
    )

    # The original creation time should remain unchanged
    for key, value in report.items():
        if key != "created_utc":
            assert existing_report.get(key) == value, (
                f"FAILED: existing report differs for {key}"
            )

    report = existing_report
    print("RERUN: existing report verified; no report overwritten.")

else:
    dbutils.fs.put(
        report_path,
        json.dumps(report, indent=2),
        overwrite=False
    )
    print("FIRST RUN: cleaning report saved.")

print(json.dumps(report, indent=2))
print(f"\nReport saved: {report_path}")

# COMMAND ----------

# Record the published files before the test
def file_snapshot(path):
    return sorted(
        (item.path, item.size, item.modificationTime)
        for item in dbutils.fs.ls(path)
    )

before_files = file_snapshot(table_location)

# Small separate test input: deliberately missing "rating"
bad_input = spark.createDataFrame(
    [("1", "10", "1", "0")],
    ["user_id", "book_id", "is_read", "is_reviewed"]
)

# The production column check must reject it
try:
    validate_columns(bad_input)
except ValueError as error:
    print(f"EXPECTED FAILURE: {error}")
else:
    raise AssertionError("FAILED: bad input was accepted")

# Confirm published files were not changed
after_files = file_snapshot(table_location)

assert before_files == after_files, (
    "FAILED: published files changed"
)

print("PASS: missing column rejected.")
print("PASS: published files unchanged.")

# COMMAND ----------

table_name = (
    "dsai3202_databricks.proj_bookrating.bookrating_ratings_v1"
)

spark.sql(f"""
    COMMENT ON TABLE {table_name} IS
    'Goodreads v1 explicit ratings. Source: raw Goodreads interactions CSV.
    One row per user-book pair. Ratings of 0 are unrated and excluded.
    IDs and indicator fields validated; no duplicate pairs found.
    Stored as Snappy-compressed Parquet. No train/test split applied yet.'
""")

column_comments = {
    "user_id": (
        "Nonnegative reader identifier. Together with book_id, "
        "forms the logical unique key."
    ),
    "book_id": (
        "Nonnegative book identifier. Together with user_id, "
        "forms the logical unique key."
    ),
    "is_read": (
        "Source read indicator: 0 or 1. Retained for reference."
    ),
    "rating": (
        "Explicit whole-star rating from 1 to 5. Prediction target; "
        "must not be included directly in model input features."
    ),
    "is_reviewed": (
        "Source review indicator: 0 or 1. Retained for reference."
    )
}

for column, description in column_comments.items():
    spark.sql(f"""
        ALTER TABLE {table_name}
        ALTER COLUMN {column} COMMENT '{description}'
    """)

display(spark.sql(f"DESCRIBE TABLE {table_name}"))

print("PASS: table and column descriptions saved.")

# COMMAND ----------

