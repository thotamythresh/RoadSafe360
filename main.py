
# # # from pathlib import Path
# # # import json

# # # import joblib
# # # import pandas as pd

# # # from fastapi import FastAPI, HTTPException
# # # from pydantic import BaseModel, ConfigDict


# # # # --------------------------------
# # # # 1. Initialize FastAPI
# # # # --------------------------------

# # # app = FastAPI(
# # #     title="RoadSafe 360 API",
# # #     description="Road Accident Severity Prediction API",
# # #     version="1.0.0"
# # # )


# # # # --------------------------------
# # # # 2. Load trained ML model
# # # # --------------------------------

# # # BASE_DIR = Path(__file__).resolve().parent

# # # MODEL_PATH = BASE_DIR / "roadsafe360_model.pkl"
# # # METADATA_PATH = BASE_DIR / "roadsafe360_metadata.json"

# # # model = joblib.load(MODEL_PATH)

# # # with open(METADATA_PATH, "r") as file:
# # #     metadata = json.load(file)

# # # FEATURES = metadata["features"]


# # # # --------------------------------
# # # # 3. API request schema
# # # # --------------------------------

# # # class AccidentInput(BaseModel):

# # #     model_config = ConfigDict(extra="forbid")

# # #     features: dict


# # # # --------------------------------
# # # # 4. Home endpoint
# # # # --------------------------------

# # # @app.get("/")
# # # def home():

# # #     return {
# # #         "message": "Welcome to RoadSafe 360 API",
# # #         "status": "Running",
# # #         "model": metadata["model_name"]
# # #     }


# # # # --------------------------------
# # # # 5. Health endpoint
# # # # --------------------------------

# # # @app.get("/health")
# # # def health():

# # #     return {
# # #         "status": "healthy",
# # #         "model_loaded": model is not None
# # #     }


# # # # --------------------------------
# # # # 6. Prediction endpoint
# # # # --------------------------------

# # # @app.post("/predict")
# # # def predict_accident(data: AccidentInput):

# # #     try:

# # #         input_features = data.features

# # #         missing_features = [
# # #             feature for feature in FEATURES
# # #             if feature not in input_features
# # #         ]

# # #         extra_features = [
# # #             feature for feature in input_features
# # #             if feature not in FEATURES
# # #         ]

# # #         if missing_features or extra_features:

# # #             raise HTTPException(
# # #                 status_code=422,
# # #                 detail={
# # #                     "missing_features": missing_features,
# # #                     "extra_features": extra_features
# # #                 }
# # #             )

# # #         input_df = pd.DataFrame(
# # #             [input_features],
# # #             columns=FEATURES
# # #         )

# # #         prediction = model.predict(input_df)

# # #         severity = int(prediction[0])

# # #         return {
# # #             "project": "RoadSafe 360",
# # #             "prediction": {
# # #                 "severity": severity
# # #             },
# # #             "message": f"Predicted accident severity: {severity}"
# # #         }

# # #     except HTTPException:
# # #         raise

# # #     except Exception as error:

# # #         raise HTTPException(
# # #             status_code=500,
# # #             detail=str(error)
# # #         )

# # from pathlib import Path
# # import json
# # import os

# # import joblib
# # import pandas as pd

# # from huggingface_hub import hf_hub_download
# # from fastapi import FastAPI, HTTPException
# # from pydantic import BaseModel, ConfigDict


# # # --------------------------------
# # # 1. Initialize FastAPI
# # # --------------------------------

# # app = FastAPI(
# #     title="RoadSafe 360 API",
# #     description="Road Accident Severity Prediction API",
# #     version="1.0.0"
# # )


# # # --------------------------------
# # # 2. Load metadata
# # # --------------------------------

# # BASE_DIR = Path(__file__).resolve().parent

# # METADATA_PATH = BASE_DIR / "roadsafe360_metadata.json"

# # with open(METADATA_PATH, "r") as file:
# #     metadata = json.load(file)

# # FEATURES = metadata["features"]


# # # --------------------------------
# # # 3. Download model from Hugging Face
# # # --------------------------------

# # REPO_ID = "mythresh9/roadsafe360"
# # MODEL_FILENAME = "roadsafe360_model.pkl"

# # HF_TOKEN = os.getenv("HF_TOKEN")

# # print("Downloading RoadSafe 360 model from Hugging Face...")

# # MODEL_PATH = hf_hub_download(
# #     repo_id=REPO_ID,
# #     filename=MODEL_FILENAME,
# #     token=HF_TOKEN
# # )

# # print("Model downloaded successfully!")
# # print("Loading trained model...")

# # model = joblib.load(MODEL_PATH)

# # print("RoadSafe 360 model loaded successfully!")


# # # --------------------------------
# # # 4. API request schema
# # # --------------------------------

# # class AccidentInput(BaseModel):

# #     model_config = ConfigDict(extra="forbid")

# #     features: dict


# # # --------------------------------
# # # 5. Home endpoint
# # # --------------------------------

# # @app.get("/")
# # def home():

# #     return {
# #         "message": "Welcome to RoadSafe 360 API",
# #         "status": "Running",
# #         "model": metadata["model_name"]
# #     }


# # # --------------------------------
# # # 6. Health endpoint
# # # --------------------------------

# # @app.get("/health")
# # def health():

# #     return {
# #         "status": "healthy",
# #         "model_loaded": model is not None
# #     }


# # # --------------------------------
# # # 7. Prediction endpoint
# # # --------------------------------

# # @app.post("/predict")
# # def predict_accident(data: AccidentInput):

# #     try:

# #         input_features = data.features

# #         missing_features = [
# #             feature for feature in FEATURES
# #             if feature not in input_features
# #         ]

# #         extra_features = [
# #             feature for feature in input_features
# #             if feature not in FEATURES
# #         ]

# #         if missing_features or extra_features:

# #             raise HTTPException(
# #                 status_code=422,
# #                 detail={
# #                     "missing_features": missing_features,
# #                     "extra_features": extra_features
# #                 }
# #             )

# #         input_df = pd.DataFrame(
# #             [input_features],
# #             columns=FEATURES
# #         )

# #         prediction = model.predict(input_df)

# #         severity = int(prediction[0])

# #         return {
# #             "project": "RoadSafe 360",
# #             "prediction": {
# #                 "severity": severity
# #             },
# #             "message": f"Predicted accident severity: {severity}"
# #         }

# #     except HTTPException:
# #         raise

# #     except Exception as error:

# #         raise HTTPException(
# #             status_code=500,
# #             detail=str(error)
# #         )

# from pathlib import Path
# import json
# import os

# import joblib
# import pandas as pd

# from huggingface_hub import hf_hub_download
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel, ConfigDict


# # ==========================================
# # 1. FASTAPI APPLICATION
# # ==========================================

# app = FastAPI(
#     title="RoadSafe 360 API",
#     description="Road Accident Severity Prediction API",
#     version="1.0.0"
# )


# # ==========================================
# # 2. LOAD PROJECT METADATA
# # ==========================================

# BASE_DIR = Path(__file__).resolve().parent

# METADATA_PATH = BASE_DIR / "roadsafe360_metadata.json"

# with open(METADATA_PATH, "r", encoding="utf-8") as file:
#     metadata = json.load(file)

# FEATURES = metadata["features"]


# # ==========================================
# # 3. HUGGING FACE MODEL CONFIGURATION
# # ==========================================

# REPO_ID = "mythresh9/roadsafe360"

# MODEL_FILENAME = "roadsafe360_model_compressed.pkl"

# HF_TOKEN = os.getenv("HF_TOKEN")


# # ==========================================
# # 4. DOWNLOAD AND LOAD TRAINED MODEL
# # ==========================================

# print("Downloading RoadSafe 360 model from Hugging Face...")

# MODEL_PATH = hf_hub_download(
#     repo_id=REPO_ID,
#     filename=MODEL_FILENAME,
#     token=HF_TOKEN
# )

# print("Model downloaded successfully!")

# print("Loading trained model...")

# model = joblib.load(MODEL_PATH)

# print("RoadSafe 360 model loaded successfully!")


# # ==========================================
# # 5. INPUT VALIDATION
# # ==========================================

# class AccidentInput(BaseModel):

#     model_config = ConfigDict(extra="forbid")

#     features: dict


# # ==========================================
# # 6. HOME ENDPOINT
# # ==========================================

# @app.get("/")
# def home():

#     return {
#         "project": "RoadSafe 360",
#         "message": "Welcome to RoadSafe 360 API",
#         "status": "Running",
#         "model": metadata["model_name"]
#     }


# # ==========================================
# # 7. HEALTH CHECK ENDPOINT
# # ==========================================

# @app.get("/health")
# def health():

#     return {
#         "status": "healthy",
#         "model_loaded": model is not None
#     }


# # ==========================================
# # 8. ACCIDENT SEVERITY PREDICTION
# # ==========================================

# @app.post("/predict")
# def predict_accident(data: AccidentInput):

#     try:

#         # Get input features
#         input_features = data.features

#         # Check missing features
#         missing_features = [
#             feature
#             for feature in FEATURES
#             if feature not in input_features
#         ]

#         # Check unexpected features
#         extra_features = [
#             feature
#             for feature in input_features
#             if feature not in FEATURES
#         ]

#         # Validate feature names
#         if missing_features or extra_features:

#             raise HTTPException(
#                 status_code=422,
#                 detail={
#                     "missing_features": missing_features,
#                     "extra_features": extra_features
#                 }
#             )

#         # Convert input into DataFrame
#         input_df = pd.DataFrame(
#             [input_features],
#             columns=FEATURES
#         )

#         # Predict accident severity
#         prediction = model.predict(input_df)

#         severity = int(prediction[0])

#         # Return prediction result
#         return {
#             "project": "RoadSafe 360",
#             "prediction": {
#                 "severity": severity
#             },
#             "message": f"Predicted accident severity: {severity}"
#         }

#     except HTTPException:

#         raise

#     except Exception as error:

#         raise HTTPException(
#             status_code=500,
#             detail=str(error)
#         )

from pathlib import Path
import json
import os

import joblib
import pandas as pd
import sklearn

from huggingface_hub import hf_hub_download
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict


# ==========================================
# 1. FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="RoadSafe 360 API",
    description="Road Accident Severity Prediction API",
    version="1.0.0"
)


# ==========================================
# 2. LOAD PROJECT METADATA
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

METADATA_PATH = BASE_DIR / "roadsafe360_metadata.json"

if not METADATA_PATH.exists():
    raise FileNotFoundError(
        f"Metadata file not found: {METADATA_PATH}"
    )

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

FEATURES = metadata["features"]

if not FEATURES:
    raise ValueError("No features found in metadata file.")


# ==========================================
# 3. HUGGING FACE MODEL CONFIGURATION
# ==========================================

REPO_ID = "mythresh9/roadsafe360"

# Exact filename currently reported in Hugging Face
MODEL_FILENAME = "roadsafe360_model.pkl"

HF_TOKEN = os.getenv("HF_TOKEN")

print("=" * 50)
print("ROADSAFE 360 - MODEL INITIALIZATION")
print("=" * 50)

print(f"Scikit-learn version: {sklearn.__version__}")
print(f"Hugging Face repository: {REPO_ID}")
print(f"Model filename: {MODEL_FILENAME}")


# ==========================================
# 4. DOWNLOAD MODEL FROM HUGGING FACE
# ==========================================

print("\nDownloading RoadSafe 360 model...")

try:
    MODEL_PATH = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILENAME,
        token=HF_TOKEN
    )

    print("Model downloaded successfully!")

except Exception as error:
    print(f"Model download failed: {error}")
    raise


# ==========================================
# 5. LOAD TRAINED MODEL
# ==========================================

print("\nLoading trained model...")

try:
    model = joblib.load(MODEL_PATH)

    print("RoadSafe 360 model loaded successfully!")

except Exception as error:
    print(f"Model loading failed: {error}")
    raise


# ==========================================
# 6. INPUT VALIDATION
# ==========================================

class AccidentInput(BaseModel):

    model_config = ConfigDict(extra="forbid")

    features: dict


# ==========================================
# 7. HOME ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {
        "project": "RoadSafe 360",
        "message": "Welcome to RoadSafe 360 API",
        "status": "Running",
        "model": metadata.get(
            "model_name",
            MODEL_FILENAME
        ),
        "model_loaded": model is not None
    }


# ==========================================
# 8. HEALTH CHECK ENDPOINT
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_file": MODEL_FILENAME,
        "sklearn_version": sklearn.__version__
    }


# ==========================================
# 9. ACCIDENT SEVERITY PREDICTION
# ==========================================

@app.post("/predict")
def predict_accident(data: AccidentInput):

    try:

        # Get input features
        input_features = data.features

        # Check missing features
        missing_features = [
            feature
            for feature in FEATURES
            if feature not in input_features
        ]

        # Check unexpected features
        extra_features = [
            feature
            for feature in input_features
            if feature not in FEATURES
        ]

        # Reject invalid input
        if missing_features or extra_features:

            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Invalid input features",
                    "missing_features": missing_features,
                    "extra_features": extra_features
                }
            )

        # Arrange features in the correct order
        input_df = pd.DataFrame(
            [input_features],
            columns=FEATURES
        )

        # Predict accident severity
        prediction = model.predict(input_df)

        severity = int(prediction[0])

        # Return prediction result
        return {
            "project": "RoadSafe 360",
            "prediction": {
                "severity": severity
            },
            "message": (
                f"Predicted accident severity: {severity}"
            )
        }

    except HTTPException:
        raise

    except Exception as error:

        print(f"Prediction error: {error}")

        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during prediction."
        )
