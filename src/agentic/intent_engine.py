# src/agentic/intent_engine.py
from typing import Dict, Any, List, Tuple

# Thresholds for SHAP-based decisions
SHAP_POSITIVE_THRESHOLD = 0.02
SHAP_NEGATIVE_THRESHOLD = -0.02

# Intent rules: (feature, predicate_fn, intent_label, priority_score)
INTENT_RULES = [
    ("lead_time_days", lambda v: v is not None and v <= 2, "late_booking", 0.90),
    ("price_sensitivity", lambda v: v is not None and v >= 0.7, "value_sensitive", 0.80),
    ("price_sensitivity", lambda v: v is not None and v <= 0.3, "low_price_sensitivity", 0.60),
    ("stay_nights", lambda v: v is not None and v <= 2, "short_stay", 0.85),
    ("stay_nights", lambda v: v is not None and v >= 7, "long_stay", 0.70),
    ("quality_expectations", lambda v: v is not None and v >= 0.7, "high_expectations", 0.95),
    ("quality_expectations", lambda v: v is not None and v <= 0.3, "low_expectations", 0.30),
    ("cluster", lambda v: v == 0, "premium_cluster", 1.00),
    ("cluster", lambda v: v == 1, "value_cluster", 0.50),
    ("travel_frequency", lambda v: v is not None and v >= 0.7, "frequent_traveler", 0.65),
    ("loyalty_propensity", lambda v: v is not None and v >= 0.6, "loyal_guest", 0.75),
]

def interpret_shap(shap_contribs: Dict[str, float], feature_values: Dict[str, Any]) -> List[Tuple[str, float]]:
    """
    Convert SHAP contributions + feature-values into ranked intents.
    Returns list of (intent_label, score) sorted by score desc.
    """
    intents: Dict[str, float] = {}

    # 1) SHAP-driven intents
    for feat, shap_val in (shap_contribs or {}).items():
        if shap_val is None:
            continue
        if shap_val <= SHAP_NEGATIVE_THRESHOLD:
            if feat == "lead_time_days":
                intents["late_booking"] = max(intents.get("late_booking", 0.0), abs(shap_val))
            if feat == "price_sensitivity":
                intents["value_sensitive"] = max(intents.get("value_sensitive", 0.0), abs(shap_val))
            if feat == "stay_nights":
                intents["short_stay"] = max(intents.get("short_stay", 0.0), abs(shap_val))
            if feat == "quality_expectations":
                intents["high_expectations"] = max(intents.get("high_expectations", 0.0), abs(shap_val))
        elif shap_val >= SHAP_POSITIVE_THRESHOLD:
            if feat == "price_per_night":
                intents["low_price_sensitivity"] = max(intents.get("low_price_sensitivity", 0.0), shap_val)
            if feat == "loyalty_propensity":
                intents["loyal_guest"] = max(intents.get("loyal_guest", 0.0), shap_val)
            if feat == "travel_frequency":
                intents["frequent_traveler"] = max(intents.get("frequent_traveler", 0.0), shap_val)

    # 2) Rule-based (feature value driven)
    for feat, fn, label, priority in INTENT_RULES:
        try:
            val = feature_values.get(feat)
            if fn(val):
                current = intents.get(label, 0.0)
                intents[label] = max(current, priority)
        except Exception:
            continue

    ranked = sorted(intents.items(), key=lambda kv: kv[1], reverse=True)
    return ranked
