# src/agentic/email_agent.py
"""
High-level Agentic Email Orchestrator
-------------------------------------
This version includes AUTOMATIC CLUSTER DETECTION from your trained K-Means model.
No manual cluster input is needed from the user.

Flow:
1. Compute cluster using KMeans (same features as training)
2. Predict expected satisfaction using LightGBM
3. Use SHAP to identify guest intents
4. Convert intents → interventions
5. Render luxury / friendly tone email using renderer_v3
"""

from typing import Dict, Any
from pathlib import Path
import numpy as np
import joblib

from .explain_api import explain_booking
from .intent_engine import interpret_shap
from .intervention_engine import map_intents_to_interventions
from .renderer import generate_agentic_email_v3  # use the luxury renderer v3

# ---------------------------------------------------------
# LOAD CLUSTERING MODELS (KMeans + Scaler)
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]   # project root
MODEL_DIR = ROOT_DIR / "models"

KMEANS_MODEL = joblib.load(MODEL_DIR / "kmeans_model.pkl")
KMEANS_SCALER = joblib.load(MODEL_DIR / "kmeans_scaler.pkl")

# Features used during clustering training
CLUSTER_FEATURES = [
  "stay_nights",
  "lead_time_days",
  "price_per_night",
  "is_weekend_stay",
  "number_of_guests",
  "checkin_month",
  "age",
  "price_sensitivity",
  "quality_expectations",
  "travel_frequency",
  "loyalty_propensity"
]



# ---------------------------------------------------------
#             AUTO CLUSTER PREDICTION
# ---------------------------------------------------------

def predict_cluster(booking: Dict[str, Any]) -> int:
    """
    Computes the guest's cluster using the trained KMeans model.
    No manual values required.
    """
    row = []

    for col in CLUSTER_FEATURES:
        row.append(float(booking.get(col, 0)))

    X = np.array([row], dtype=float)
    X_scaled = KMEANS_SCALER.transform(X)

    cluster = int(KMEANS_MODEL.predict(X_scaled)[0])
    return cluster


# ---------------------------------------------------------
#                  MAIN ORCHESTRATION
# ---------------------------------------------------------

def generate_email_for_booking(
    booking: Dict[str, Any],
    hotel_context: Dict[str, Any] = None,
    guest_name: str = "Guest",
    hotel_name: str = "Cinnamon Hotels"
) -> Dict[str, Any]:
    """
    Master function: auto-cluster → predict → explain → generate email.
    """

    hotel_context = hotel_context or {}

    # 1) Auto compute cluster
    cluster = predict_cluster(booking)
    booking["cluster"] = cluster  # IMPORTANT

    # 2) Predict expected rating + SHAP explanation
    model_output = explain_booking(booking)
    prediction = model_output["prediction"]
    shap_contribs = model_output["shap_contribs"]

    # 3) Intent extraction
    intents = interpret_shap(shap_contribs, model_output["input_features"])

    # 4) Intervention generation
    interventions = map_intents_to_interventions(intents, hotel_context)

    # 5) Render final email with luxury tone v3
    email_result = generate_agentic_email_v3(
        booking=booking,
        shap_contribs=shap_contribs,
        model_prediction=prediction,
        hotel_context=hotel_context,
        guest_name=guest_name,
        hotel_name=hotel_name
    )

    # Attach final metadata
    email_result["prediction"] = prediction
    email_result["cluster"] = cluster

    return email_result


# ---------------------------------------------------------
#                    LOCAL TESTING
# ---------------------------------------------------------
if __name__ == "__main__":
    sample_booking = {
        "booking_id": "text123",

        
        "lead_time_days": 2,           
        "stay_nights": 1,              
        "price_per_night": 95,         
        "number_of_guests": 1,

       
        "price_sensitivity": 0.85,     
        "quality_expectations": 0.35,  

        
        "travel_frequency": 1,         
        "loyalty_propensity": 0.15,    
        "age": 24,                     
    }


    hotel_context = {
        "offers": [
            "10% off dinner buffet",
            "Complimentary Wi-Fi",
            "Free welcome drink"
        ],
        "local_events": [
            "Night Market",
            "Street Food Festival"
        ]
    }



    result = generate_email_for_booking(sample_booking, hotel_context, guest_name="Alex", hotel_name="Cinnamon Grand")

    print("\n--- EMAIL PREVIEW (TEST) ---")
    print(result["email"]["plain"])
    print("\nCluster detected:", result["cluster"])
    print("Prediction:", result["prediction"])
