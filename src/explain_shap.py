# src/explain_shap.py
import os
import joblib
import json
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from typing import Dict, Any

# Paths
MODELS_DIR = "models"
DATA_PATH = "data/bookings_with_clusters.csv"
FEATURES_FILE = os.path.join(MODELS_DIR, "feature_columns_used.pkl")
BEST_MODEL_FILE = os.path.join(MODELS_DIR, "best_model.pkl")  # not required but useful
OUTPUT_DIR = "explain_outputs"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
SHAP_DIR = os.path.join(OUTPUT_DIR, "shap_values")
HTML_DIR = os.path.join(OUTPUT_DIR, "html")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(SHAP_DIR, exist_ok=True)
os.makedirs(HTML_DIR, exist_ok=True)

# Models we will explain (filenames created in training step)
MODEL_FILES = {
    "LightGBM": os.path.join(MODELS_DIR, "LightGBM_pipelineA.pkl"),
    "XGBoost": os.path.join(MODELS_DIR, "XGBoost_pipelineA.pkl"),
    "CatBoost": os.path.join(MODELS_DIR, "CatBoost_pipelineA.pkl"),
}

# ---- Utility: load feature columns ----
if not os.path.exists(FEATURES_FILE):
    raise FileNotFoundError(f"Feature columns file not found: {FEATURES_FILE}")

feature_cols = joblib.load(FEATURES_FILE)
print(f"Loaded feature columns ({len(feature_cols)}): {feature_cols}")

# ---- Load dataset (for background) ----
df = pd.read_csv(DATA_PATH)
# keep only rows with ratings (for later sampling)
df_targeted = df.dropna(subset=["rating"]).reset_index(drop=True)

# helper to ensure column order & presence
def make_X_matrix(df_like: pd.DataFrame) -> pd.DataFrame:
    # ensure all expected columns exist in df_like, else create with NaN
    for c in feature_cols:
        if c not in df_like.columns:
            df_like[c] = np.nan
    X = df_like[feature_cols].copy()
    return X

# ---- Load models ----
models = {}
for name, path in MODEL_FILES.items():
    if not os.path.exists(path):
        print(f"Warning: model file not found for {name}: {path} — skipping this model.")
        continue
    models[name] = joblib.load(path)
print("Loaded models:", list(models.keys()))

# ---- Build SHAP explainers and compute SHAP values ----
explainers = {}
shap_values_store = {}  # name -> shap_values array
base_values = {}

# We will compute SHAP on the full training/background X (using df_targeted)
X_background = make_X_matrix(df_targeted)  # many rows; shap will sample internally if needed

# Some boosters (CatBoost) may require passing np.array; TreeExplainer will handle scikit-learn wrappers
for name, model in models.items():
    print(f"\nBuilding SHAP explainer for {name} ...")
    # Use TreeExplainer (fast for tree models)
    explainer = shap.TreeExplainer(model)
    explainers[name] = explainer

    # Calculate SHAP values for the background dataset
    print(f"  Computing SHAP values for background data (this may take a bit)...")
    shap_vals = explainer.shap_values(X_background)  # for regression this is usually array shape (n_rows, n_features)
    # Normalize shape: some libraries return list-of-arrays for multiclass; here regression -> single array
    shap_values_store[name] = shap_vals
    # save numeric shap values
    np.save(os.path.join(SHAP_DIR, f"shap_values_{name}.npy"), shap_vals)
    print(f"  Saved SHAP numpy -> {os.path.join(SHAP_DIR, f'shap_values_{name}.npy')}")

    # Base value (expected value)
    try:
        base_val = float(explainer.expected_value)
    except Exception:
        # fallback
        base_val = float(np.mean(model.predict(X_background)))
    base_values[name] = base_val
    print(f"  Base value for {name}: {base_val}")

    # ---- Global feature importance: mean absolute SHAP ----
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    feat_imp = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=False)
    feat_imp.to_csv(os.path.join(OUTPUT_DIR, f"feature_importance_{name}.csv"), index=False)
    print(f"  Saved feature importance CSV -> {os.path.join(OUTPUT_DIR, f'feature_importance_{name}.csv')}")

    # ---- Summary plot (matplotlib) ----
    plt.figure(figsize=(10, 6))
    # shap.summary_plot expects SHAP values and the data; use show=False / then save figure.
    try:
        shap.summary_plot(shap_vals, X_background, show=False)
        plt.title(f"SHAP Summary Plot ({name})")
        png_path = os.path.join(PLOTS_DIR, f"shap_summary_{name}.png")
        plt.savefig(png_path, bbox_inches="tight", dpi=250)
        plt.close()
        print(f"  Saved SHAP summary PNG -> {png_path}")
    except Exception as e:
        print(f"  Could not create summary plot for {name}: {e}")

    # ---- Also create a bar plot (mean abs shap) ----
    try:
        plt.figure(figsize=(8, 6))
        topn = min(20, len(feature_cols))
        top_feat = feat_imp.head(topn).iloc[::-1]
        plt.barh(top_feat["feature"], top_feat["mean_abs_shap"])
        plt.xlabel("Mean absolute SHAP value")
        plt.title(f"Top {topn} Feature Importance (mean |SHAP|) - {name}")
        bar_png = os.path.join(PLOTS_DIR, f"shap_bar_{name}.png")
        plt.savefig(bar_png, bbox_inches="tight", dpi=250)
        plt.close()
        print(f"  Saved SHAP bar PNG -> {bar_png}")
    except Exception as e:
        print(f"  Could not create bar plot for {name}: {e}")

    # ---- Save an interactive HTML force plot for one representative sample (the median row) ----
    try:
        sample_idx = max(0, X_background.shape[0] // 2)
        sample_X = X_background.iloc[[sample_idx]]
        sample_shap = explainer.shap_values(sample_X)
        # Create force plot object (JS)
        force_html = shap.force_plot(base_val, sample_shap[0] if isinstance(sample_shap, np.ndarray) else sample_shap, sample_X, matplotlib=False)
        html_path = os.path.join(HTML_DIR, f"shap_force_{name}.html")
        shap.save_html(html_path, force_html)
        print(f"  Saved interactive force plot HTML -> {html_path}")
    except Exception as e:
        print(f"  Could not save interactive HTML force plot for {name}: {e}")

print("\nSHAP artifacts generation completed. Files are in 'explain_outputs/'.")

# ---- Helper function: explain a single booking dict ----
def explain_single_booking(booking: Dict[str, Any], model_names=None, save_individual_html=True) -> Dict[str, Any]:
    """
    booking: dict with keys equal to feature columns (feature_cols). Missing keys will be treated as NaN.
    model_names: list of model names to explain (subset of models.keys()). If None, explain all loaded models.
    save_individual_html: whether to save a per-booking HTML force plot (per model)
    Returns a JSON-serializable dict with predictions, base values, top contributors.
    """
    if model_names is None:
        model_names = list(models.keys())
    out = {"booking": {}, "explanations": {}}

    # Build DataFrame for the booking
    booking_df = pd.DataFrame([booking])
    X_row = make_X_matrix(booking_df)

    # Save the booking as CSV for record (optional)
    booking_id = booking.get("booking_id", "manual_sample")
    booking_csv_path = os.path.join(OUTPUT_DIR, f"booking_{booking_id}.csv")
    X_row.to_csv(booking_csv_path, index=False)

    out["booking"]["csv"] = booking_csv_path

    for name in model_names:
        if name not in models:
            continue
        model = models[name]
        explainer = explainers[name]
        base_val = base_values.get(name, None)

        # Predict
        # For boosting pipeline we used numeric-only columns during training (boost_numeric_cols)
        # but explainer was computed on full feature_cols; we will pass the full X_row as created earlier.
        pred = float(model.predict(X_row[feature_cols])[0]) if hasattr(model, "predict") else float(model.predict(X_row)[0])

        # SHAP for this single row
        try:
            shap_val = explainer.shap_values(X_row)  # shape (1, n_features)
            if isinstance(shap_val, list):
                shap_val_arr = np.array(shap_val)[0]
            else:
                shap_val_arr = shap_val[0] if shap_val.ndim == 3 else shap_val[0] if shap_val.shape[0] == 1 else shap_val
        except Exception:
            # Fallback: compute explainer(X_row) with new API
            explanation = explainer(X_row)
            shap_val_arr = explanation.values[0]

        # create DataFrame of contributions
        contrib = pd.DataFrame({
            "feature": feature_cols,
            "shap": shap_val_arr,
            "abs_shap": np.abs(shap_val_arr),
            "feature_value": [X_row[c].iloc[0] for c in feature_cols]
        }).sort_values("shap", ascending=False)

        top_pos = contrib[contrib["shap"] > 0].head(5).to_dict(orient="records")
        top_neg = contrib[contrib["shap"] < 0].head(5).to_dict(orient="records")

        explanation = {
            "prediction": pred,
            "base_value": base_val,
            "top_positive_contributors": top_pos,
            "top_negative_contributors": top_neg,
            "all_contributions_csv": os.path.join(OUTPUT_DIR, f"booking_{booking_id}_{name}_contribs.csv")
        }

        # save contributions CSV
        contrib.to_csv(explanation["all_contributions_csv"], index=False)

        # optional: save a per-booking force plot HTML
        if save_individual_html:
            try:
                fp = shap.force_plot(base_val, shap_val_arr, X_row, matplotlib=False)
                html_path = os.path.join(HTML_DIR, f"booking_{booking_id}_force_{name}.html")
                shap.save_html(html_path, fp)
                explanation["force_plot_html"] = html_path
            except Exception as e:
                explanation["force_plot_html_error"] = str(e)

        out["explanations"][name] = explanation

    # Save final JSON
    out_json_path = os.path.join(OUTPUT_DIR, f"booking_{booking_id}_explanation.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    out["json"] = out_json_path
    return out

# ---- Example usage: explain a random booking from the dataset ----
if __name__ == "__main__":
    # pick an index to inspect
    sample_idx = 0
    sample_row = df_targeted.iloc[[sample_idx]]
    booking_dict = {c: sample_row[c].iloc[0] for c in sample_row.columns if c in feature_cols or c in ["booking_id"]}
    print("\nExample booking features (first sample):")
    print({k: booking_dict.get(k) for k in feature_cols})

    explanation = explain_single_booking(booking_dict, model_names=list(models.keys()), save_individual_html=True)
    print("\nSaved per-booking explanation (JSON):", explanation["json"])
    for m in explanation["explanations"]:
        print(f"  - {m}: prediction={explanation['explanations'][m]['prediction']}, base={explanation['explanations'][m]['base_value']}")
