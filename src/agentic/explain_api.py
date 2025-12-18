# src/agentic/explain_api.py
"""
Inference + SHAP Explainability Engine
--------------------------------------
This module loads:
 - Best trained model
 - Feature list
 - SHAP explainers for LightGBM/XGBoost/CatBoost

And exposes one main function:

    explain_booking(booking_dict)

which returns:
 - prediction
 - shap_contribs (dict feature → shap value)
 - cleaned booking feature vector
 - model name used
 - optional raw SHAP array for debugging/logs
"""

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

# Always reference models folder from project root
ROOT_DIR = Path(__file__).resolve().parents[2]   # goes up to project root
MODEL_DIR = ROOT_DIR / "models"

BEST_MODEL_FILE = MODEL_DIR / "best_model.pkl"
FEATURES_FILE = MODEL_DIR / "feature_columns_used.pkl"

# Optional model-specific SHAP explainers
SHAP_DIR = "explain_outputs/shap_values"  # earlier scripts saved here but we recompute if needed

# Load feature list
if not os.path.exists(FEATURES_FILE):
    raise FileNotFoundError("Missing feature_columns_used.pkl")

FEATURE_COLS = joblib.load(FEATURES_FILE)

# Safe load model
if not os.path.exists(BEST_MODEL_FILE):
    raise FileNotFoundError("Missing best_model.pkl")
BEST_MODEL = joblib.load(BEST_MODEL_FILE)

# Identify model type for SHAP
MODEL_NAME = BEST_MODEL.__class__.__name__.lower()

# Lazily-loaded SHAP explainer
_SHAP_EXPLAINER = None

# -------------------------------------------------------------------
# Utility: Ensure booking_dict → feature vector
# -------------------------------------------------------------------
def make_feature_row(booking: Dict[str, Any]) -> pd.DataFrame:
    """
    Converts incoming booking dict → DataFrame with all required FEATURE_COLS.
    Missing values become np.nan.
    Unknown extra fields are ignored.
    """
    row = {}
    for col in FEATURE_COLS:
        row[col] = booking.get(col, np.nan)

    return pd.DataFrame([row])


# -------------------------------------------------------------------
# Initialize SHAP explainer lazily
# -------------------------------------------------------------------
def _load_shap_explainer():
    """
    Use shap.TreeExplainer for boosting models.
    Only computed once — cached after first call.
    """
    global _SHAP_EXPLAINER

    if _SHAP_EXPLAINER is not None:
        return _SHAP_EXPLAINER

    try:
        import shap
        # Fast path for tree models
        _SHAP_EXPLAINER = shap.TreeExplainer(BEST_MODEL)
        return _SHAP_EXPLAINER
    except Exception as e:
        print("⚠️ Warning: Could not initialize SHAP explainer:", e)
        _SHAP_EXPLAINER = None
        return None


# -------------------------------------------------------------------
# Main API function
# -------------------------------------------------------------------
def explain_booking(booking: Dict[str, Any]) -> Dict[str, Any]:
    """
    booking: dict containing raw features (cluster, lead_time_days, etc.)

    Returns:
        {
            "prediction": float,
            "shap_contribs": { feature_name: shap_value },
            "model": model_name,
            "input_features": { feature_name: value },
            "raw_shap": numpy array (optional)
        }
    """
    # Prepare input row
    X_row = make_feature_row(booking)

    # Predict rating
    try:
        pred = float(BEST_MODEL.predict(X_row)[0])
    except Exception:
        # CatBoost sometimes needs .predict(X_row.values)
        pred = float(BEST_MODEL.predict(X_row.values)[0])

    # Build SHAP explainer
    explainer = _load_shap_explainer()

    # Default placeholders
    shap_contribs = {col: 0.0 for col in FEATURE_COLS}
    raw_shap = None

    if explainer is not None:
        try:
            shap_vals = explainer.shap_values(X_row)

            # shap returns list in multiclass; regression = array
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]

            shap_vals_row = shap_vals[0]
            raw_shap = shap_vals_row

            # Map feature → shap contribution
            shap_contribs = {
                FEATURE_COLS[i]: float(shap_vals_row[i])
                for i in range(len(FEATURE_COLS))
            }

        except Exception as e:
            print("⚠️ Warning: SHAP computation failed:", e)

    # Return structured result
    return {
        "prediction": pred,
        "shap_contribs": shap_contribs,
        "model": MODEL_NAME,
        "input_features": {c: X_row[c].iloc[0] for c in FEATURE_COLS},
        "raw_shap": raw_shap
    }


# -------------------------------------------------------------------
# Optional manual test
# -------------------------------------------------------------------
if __name__ == "__main__":
    sample = {
        "lead_time_days": 1,
        "stay_nights": 2,
        "price_sensitivity": 0.5,
        "cluster": 0,
        "quality_expectations": 0.5,
        "age": 32,
        "price_per_night": 120
    }

    result = explain_booking(sample)
    print("\n--- EXPLANATION OUTPUT ---")
    print("Prediction:", result["prediction"])
    print("Top SHAP contributors:", sorted(result["shap_contribs"].items(), key=lambda x: abs(x[1]), reverse=True)[:5])
