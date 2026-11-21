import os
import json
import pandas as pd
import requests

# Read the fixed inputs (tab-separated)
df = pd.read_csv("fixed_inputs_v1.csv", sep="\t")
df.columns = df.columns.str.strip()

# 1. Prepare the payload (only the 4 features and row_id, no user_id or book_id)
inputs = []
for index, row in df.iterrows():
    inputs.append({
        "row_id": str(row["row_id"]).strip(),
        "reader_avg_rating": float(row["reader_avg_rating"]),
        "reader_rating_count": int(row["reader_rating_count"]),
        "book_avg_rating": float(row["book_avg_rating"]),
        "book_rating_count": int(row["book_rating_count"]),
    })
payload = {"inputs": inputs}

# 2. URL and key come from the terminal (export SCORING_URI=... and export KEY=...)
url = os.environ["SCORING_URI"]
key = os.environ["KEY"]
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {key}",
}

# 3. Send the request
print(f"Sending {len(inputs)} rows to the endpoint...")
response = requests.post(url, headers=headers, json=payload)

if response.status_code == 200:
    body = response.json()
    if isinstance(body, str):  # the endpoint returns JSON inside a string
        body = json.loads(body)
    results = body["predictions"]

    # 4. Compare with the offline predictions
    max_diff = 0.0
    all_passed = True

    print("\n--- Results ---")
    for pred in results:
        row_id = pred["row_id"]
        deployed_val = pred["predicted_rating"]
        expected_val = float(df[df["row_id"].astype(str).str.strip() == row_id]["expected_prediction"].values[0])

        diff = abs(deployed_val - expected_val)
        max_diff = max(max_diff, diff)

        status = "PASS" if diff <= 1e-6 else "FAIL"
        if status == "FAIL":
            all_passed = False

        print(f"{row_id}: Deployed={deployed_val:.10f} | Expected={expected_val:.10f} | Diff={diff:.10e} | {status}")

    print(f"\nRows compared: {len(results)}")
    print(f"Largest absolute difference: {max_diff:.10e}")
    print(f"Overall result: {'PASS' if all_passed else 'FAIL'} (Tolerance: <= 1e-6)")
else:
    print(f"Error {response.status_code}: {response.text}")
