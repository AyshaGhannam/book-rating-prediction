import os
import json
import time
import statistics
import requests

url = os.environ["SCORING_URI"]
key = os.environ["KEY"]
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}

GOOD = {
    "row_id": "r01",
    "reader_avg_rating": 3.7341772151898733,
    "reader_rating_count": 79,
    "book_avg_rating": 3.8767890479153704,
    "book_rating_count": 3214,
}


def bad(**changes):
    row = dict(GOOD)
    row.update(changes)
    return {"inputs": [row]}


def without(field):
    row = dict(GOOD)
    del row[field]
    return {"inputs": [row]}


# ---------- Test 1: invalid requests ----------
# Azure wraps an error from score.py in HTTP 424, and the script's own code
# is in the header ms-azureml-model-error-statuscode (we expect 400).
cases = [
    ("not JSON", "this is not json"),
    ("no inputs key", {}),
    ("empty inputs list", {"inputs": []}),
    ("missing book_avg_rating", without("book_avg_rating")),
    ("reader_avg_rating = 7 (out of range)", bad(reader_avg_rating=7)),
    ("book_avg_rating = 0.5 (out of range)", bad(book_avg_rating=0.5)),
    ("reader_rating_count = -1", bad(reader_rating_count=-1)),
    ("book_rating_count = 2.5 (not whole)", bad(book_rating_count=2.5)),
    ("reader_avg_rating = 'high' (text)", bad(reader_avg_rating="high")),
]

print("=== Test: invalid requests ===")
invalid_ok = True
invalid_results = []
for name, body in cases:
    data = body if isinstance(body, str) else json.dumps(body)
    r = requests.post(url, headers=headers, data=data)
    code = r.headers.get("ms-azureml-model-error-statuscode")
    passed = r.status_code == 424 and code == "400"
    invalid_ok = invalid_ok and passed
    msg = r.text.strip()[:110]
    print(f"{'PASS' if passed else 'FAIL'} | {name} | http={r.status_code} model_status={code} | {msg}")
    invalid_results.append({"case": name, "http": r.status_code, "model_status": code, "passed": passed})
print("Invalid-input overall:", "PASS" if invalid_ok else "FAIL")

# ---------- Test 2: performance ----------
WARMUP = 5
CALLS = 50
payload = json.dumps({"inputs": [GOOD]})

print("\n=== Test: performance (single row) ===")
for _ in range(WARMUP):
    requests.post(url, headers=headers, data=payload)

times = []
errors = 0
for _ in range(CALLS):
    t0 = time.perf_counter()
    r = requests.post(url, headers=headers, data=payload)
    times.append((time.perf_counter() - t0) * 1000)
    if r.status_code != 200:
        errors += 1

times.sort()
p95 = times[int(0.95 * len(times)) - 1]
print(f"calls={CALLS} warmup={WARMUP} errors={errors}")
print(f"median={statistics.median(times):.0f} ms  mean={statistics.mean(times):.0f} ms  p95={p95:.0f} ms  max={times[-1]:.0f} ms")

# batch of 20 identical rows
batch = json.dumps({"inputs": [GOOD] * 20})
t0 = time.perf_counter()
r = requests.post(url, headers=headers, data=batch)
print(f"batch of 20 rows: http={r.status_code} time={(time.perf_counter() - t0) * 1000:.0f} ms")

os.makedirs("evidence", exist_ok=True)
with open("evidence/invalid_and_performance.json", "w") as f:
    json.dump({
        "invalid": invalid_results,
        "performance": {
            "calls": CALLS, "errors": errors,
            "median_ms": statistics.median(times), "p95_ms": p95, "max_ms": times[-1],
        },
    }, f, indent=2)
print("Saved evidence/invalid_and_performance.json")
