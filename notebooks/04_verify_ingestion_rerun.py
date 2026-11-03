# Databricks notebook source
raw_dir = (
    "abfss://raw@bookrating60308393.dfs.core.windows.net/goodreads/v1/"
)
filename = "goodreads_interactions.csv"
expected_bytes = 4_318_621_741

def find_raw_file():
    matches = [f for f in dbutils.fs.ls(raw_dir) if f.name == filename]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {filename}; found {len(matches)}.")
    return matches[0]

before = find_raw_file()

# Rerun guard: skip upload when the existing file has the expected size.
if before.size != expected_bytes:
    raise RuntimeError("Existing file has an unexpected size. Stop and investigate.")

print("SKIP: Goodreads v1 already exists. No upload performed.")

after = find_raw_file()

assert (before.path, before.size, before.modificationTime) == (
    after.path, after.size, after.modificationTime
), "Raw file metadata changed."

print(f"Raw file size: {after.size:,} bytes")
print("PASS: raw file path, size, and modification time are unchanged.")