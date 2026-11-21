# Phase 2: Deploying the Book Rating Model on Azure ML

## What I deployed

| Item | Value |
|---|---|
| Model | `bookrating-rf-v1`, version 1 (scikit-learn Random Forest) |
| Endpoint | `bookrating-endpoint` |
| Deployment | `rf-deploy-v1` |
| Instance | `Standard_DS2_v2`, 1 instance, Qatar Central |
| Auth | Key |
| Commit | TODO: paste commit SHA |

## Files

- `score.py`: checks the input and returns predictions
- `deploy.yml`, `conda.yaml`: deployment and environment settings
- `MLmodel`: model description from MLflow
- `fixed_inputs_v1.csv`: 20 rows with Aicha's offline predictions
- `test_consistency.py`: compares deployed and offline predictions
- `test_invalid_and_performance.py`: bad requests and latency test

## Input

Send 4 features (not `user_id` or `book_id`):

```json
{"inputs": [{
  "row_id": "r01",
  "reader_avg_rating": 3.73,
  "reader_rating_count": 79,
  "book_avg_rating": 3.88,
  "book_rating_count": 3214
}]}
```

| Field | Rule |
|---|---|
| `row_id` | optional text, sent back in the answer |
| `reader_avg_rating`, `book_avg_rating` | number from 1.0 to 5.0 |
| `reader_rating_count`, `book_rating_count` | whole number, 0 or more |

## Output

```json
{"model_version": "bookrating-rf-v1_1",
 "predictions": [{"row_id": "r01", "predicted_rating": 3.704857192378488}]}
```

Bad input gives a message like `{"error": "reader_avg_rating must be a number between 1.0 and 5.0. Got: 7"}`.
Azure shows it as HTTP 424, with the header `ms-azureml-model-error-statuscode: 400`.
The body comes back as JSON inside a string, so use `json.loads` if `response.json()` gives a string.

## How to run the tests

```bash
export SCORING_URI=$(az ml online-endpoint show --name bookrating-endpoint --query scoring_uri -o tsv)
export KEY=$(az ml online-endpoint get-credentials --name bookrating-endpoint --query primaryKey -o tsv)
python test_consistency.py
python test_invalid_and_performance.py
```

## Test results

| Test | Result |
|---|---|
| Valid request | HTTP 200, r01 predicted 3.704857192378488 |
| Invalid requests (9 cases) | All pass: HTTP 424, model status 400, clear message |
| Offline vs deployed (20 rows, tolerance 1e-6) | PASS, largest difference 1.33e-15 |
| Performance (50 calls, 0 errors) | median 85 ms, p95 172 ms, max 178 ms |

The latency was measured from my laptop, so it includes network time.

## Cleanup and redeploy

The endpoint costs about $0.137 per hour, so I delete it when I am not using it:

```bash
az ml online-endpoint delete --name bookrating-endpoint --yes --no-wait
```

The model stays registered, so to redeploy:

```bash
az ml online-endpoint create --name bookrating-endpoint --auth-mode key
az ml online-deployment create --file deploy.yml --all-traffic
```

Then get the new key with the `get-credentials` command above and run the tests again.

## AI use

| Contributor | Tool | What I asked for | How AI helped | What I did and checked |
|---|---|---|---|---|
| Nour Ben Ayed | Claude | Help understanding the Phase 2 steps from Lab 10 and Aicha's handoff. | Explained the order of the steps and helped me when I got stuck. | Did the steps myself in Azure ML with the Azure CLI: registered the model and deployed `rf-deploy-v1` on `bookrating-endpoint`. |
| Nour Ben Ayed | Claude | Help with the input checks in the scoring script. | Suggested checks for missing fields, ratings from 1 to 5 and whole non-negative counts, and a way to set the error status code. | Used and edited the script, redeployed, and checked that a valid request returns 200 and bad requests return 424 with model status 400. |
| Nour Ben Ayed | Claude | Help with the test scripts. | Suggested the test cases and helped fix an indentation error. | Ran the tests on the live endpoint: 20 of 20 rows within 1e-6 (largest difference 1.33e-15), 9 of 9 invalid cases passed, median 85 ms and p95 172 ms. 

