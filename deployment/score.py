import json
import logging
import os
import pandas as pd
import mlflow.pyfunc

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/bookrating-rf-v1/1)
    model_path = os.path.join(os.environ.get("AZUREML_MODEL_DIR", ""), "model_files")
    
    # Sometimes Azure extracts the files directly into AZUREML_MODEL_DIR, 
    # sometimes into a subfolder. We try loading directly first.
    try:
        model = mlflow.pyfunc.load_model(model_path)
        logging.info("Model loaded from subfolder.")
    except Exception as e:
        # Fallback to the root directory
        model = mlflow.pyfunc.load_model(os.environ["AZUREML_MODEL_DIR"])
        logging.info("Model loaded from root.")

def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    """
    try:
        # 1. Parse the incoming JSON
        data = json.loads(raw_data)
        
        if "inputs" not in data or not isinstance(data["inputs"], list) or len(data["inputs"]) == 0:
            return {"error": "Request must contain a non-empty 'inputs' list."}, 400

        inputs = data["inputs"]
        rows = []
        row_ids = []

        # 2. Validate and extract features
        required_fields = ["reader_avg_rating", "reader_rating_count", "book_avg_rating", "book_rating_count"]
        
        for i, row in enumerate(inputs):
            # Check missing fields
            for field in required_fields:
                if field not in row:
                    return {"error": f"Missing required field: {field} in row index {i}"}, 400
                
            # Validate types and ranges
            r_avg = row["reader_avg_rating"]
            r_cnt = row["reader_rating_count"]
            b_avg = row["book_avg_rating"]
            b_cnt = row["book_rating_count"]

            if not isinstance(r_avg, (int, float)) or not (1.0 <= r_avg <= 5.0):
                return {"error": f"reader_avg_rating must be a number between 1.0 and 5.0. Got: {r_avg}"}, 400
            if not isinstance(b_avg, (int, float)) or not (1.0 <= b_avg <= 5.0):
                return {"error": f"book_avg_rating must be a number between 1.0 and 5.0. Got: {b_avg}"}, 400
            
            # Counts must be non-negative whole numbers (integers)
            if not isinstance(r_cnt, int) or r_cnt < 0:
                return {"error": f"reader_rating_count must be a non-negative integer. Got: {r_cnt}"}, 400
            if not isinstance(b_cnt, int) or b_cnt < 0:
                return {"error": f"book_rating_count must be a non-negative integer. Got: {b_cnt}"}, 400

            # Extract row_id if it exists (optional per handoff doc)
            row_ids.append(row.get("row_id", str(i)))
            
            # Add to rows for prediction (must match exact feature order from Aysha)
            rows.append({
                "reader_avg_rating": float(r_avg),
                "reader_rating_count": float(r_cnt),
                "book_avg_rating": float(b_avg),
                "book_rating_count": float(b_cnt)
            })

        # 3. Run prediction
        df = pd.DataFrame(rows)
        predictions = model.predict(df)

        # 4. Format output
        output = {
            "model_version": "bookrating-rf-v1_1", # Hardcoded based on our registry
            "predictions": []
        }
        
        for row_id, pred in zip(row_ids, predictions):
            output["predictions"].append({
                "row_id": row_id,
                "predicted_rating": float(pred)
            })

        # Return 200 OK with the JSON
        return output, 200

    except Exception as e:
        logging.error(f"Error during scoring: {str(e)}")
        return {"error": f"Internal server error: {str(e)}"}, 500
