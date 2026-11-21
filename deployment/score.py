import json
import os
import pandas as pd
import mlflow.pyfunc
from azureml.contrib.services.aml_request import rawhttp
from azureml.contrib.services.aml_response import AMLResponse

REQUIRED_FIELDS = ["reader_avg_rating", "reader_rating_count", "book_avg_rating", "book_rating_count"]


def init():
    global model
    base = os.environ.get("AZUREML_MODEL_DIR", "")
    model_path = os.path.join(base, "model_files")
    try:
        model = mlflow.pyfunc.load_model(model_path)
    except Exception:
        model = mlflow.pyfunc.load_model(base)


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_whole(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _reply(body, status):
    # real HTTP status code, body is JSON text
    return AMLResponse(json.dumps(body), status, json_str=True)


@rawhttp
def run(request):
    if request.method != "POST":
        return _reply({"error": "Use POST with a JSON body."}, 405)

    try:
        data = json.loads(request.get_data(as_text=True))
    except Exception:
        return _reply({"error": "Body is not valid JSON."}, 400)

    if not isinstance(data, dict) or not isinstance(data.get("inputs"), list) or len(data["inputs"]) == 0:
        return _reply({"error": "Request must contain a non-empty 'inputs' list."}, 400)

    rows, row_ids = [], []
    for i, row in enumerate(data["inputs"]):
        if not isinstance(row, dict):
            return _reply({"error": f"Row index {i} must be an object."}, 400)
        for field in REQUIRED_FIELDS:
            if field not in row:
                return _reply({"error": f"Missing required field: {field} in row index {i}"}, 400)

        r_avg, r_cnt = row["reader_avg_rating"], row["reader_rating_count"]
        b_avg, b_cnt = row["book_avg_rating"], row["book_rating_count"]

        if not _is_number(r_avg) or not (1.0 <= r_avg <= 5.0):
            return _reply({"error": f"reader_avg_rating must be a number between 1.0 and 5.0. Got: {r_avg}"}, 400)
        if not _is_number(b_avg) or not (1.0 <= b_avg <= 5.0):
            return _reply({"error": f"book_avg_rating must be a number between 1.0 and 5.0. Got: {b_avg}"}, 400)
        if not _is_whole(r_cnt) or r_cnt < 0:
            return _reply({"error": f"reader_rating_count must be a non-negative whole number. Got: {r_cnt}"}, 400)
        if not _is_whole(b_cnt) or b_cnt < 0:
            return _reply({"error": f"book_rating_count must be a non-negative whole number. Got: {b_cnt}"}, 400)

        row_ids.append(str(row.get("row_id", i)))
        rows.append({
            "reader_avg_rating": float(r_avg),
            "reader_rating_count": float(r_cnt),
            "book_avg_rating": float(b_avg),
            "book_rating_count": float(b_cnt),
        })

    try:
        preds = model.predict(pd.DataFrame(rows))
    except Exception as e:
        return _reply({"error": f"Internal server error: {e}"}, 500)

    out = {
        "model_version": "bookrating-rf-v1_1",
        "predictions": [{"row_id": r, "predicted_rating": float(p)} for r, p in zip(row_ids, preds)],
    }
    return _reply(out, 200)
