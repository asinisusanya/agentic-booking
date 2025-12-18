import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Traditional models
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

# Boosting models
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor


# ========================================
# Load dataset
# ========================================
DATA_PATH = "data/bookings_with_clusters.csv"
os.makedirs("models", exist_ok=True)

df = pd.read_csv(DATA_PATH)

TARGET_COL = "rating"

FEATURE_COLS = [
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
    "loyalty_propensity",
    "cluster",
]

df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

X = df[FEATURE_COLS]
y = df[TARGET_COL].astype(float)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========================================
# Identify numeric & categorical columns
# ========================================
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

print("Numeric columns:", num_cols)
print("Categorical columns:", cat_cols)

# ========================================
# Pipeline B (Traditional ML) 
# ========================================
numeric_transformer = SimpleImputer(strategy="median")

if len(cat_cols) > 0:
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
else:
    categorical_transformer = "drop"

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ],
    remainder="drop"
)

pipeline_b_models = {
    "LinearRegression": LinearRegression(),
    "Lasso": Lasso(alpha=0.01, max_iter=3000),
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),

    "MLPRegressor": Pipeline(steps=[
        ("scale", StandardScaler()),
        ("mlp", MLPRegressor(hidden_layer_sizes=(64,32), max_iter=500, random_state=42))
    ])
}

def make_pipeline_b(model):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

pipeline_b_wrapped = {name: make_pipeline_b(model)
                      for name, model in pipeline_b_models.items()}

# ========================================
# Pipeline A (Boosting Models) - NaN allowed
# ========================================
boost_numeric_cols = num_cols

X_train_boost = X_train[boost_numeric_cols]
X_val_boost = X_val[boost_numeric_cols]

pipeline_a_models = {
    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
    ),
    "LightGBM": lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    ),
    "CatBoost": CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=False,
    ),
}

# ========================================
# Training
# ========================================
results = []

print("\n=== PIPELINE B (Traditional Models) ===\n")
for name, model in pipeline_b_wrapped.items():
    try:
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        mse = mean_squared_error(y_val, preds)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_val, preds)

        print(f"{name} → RMSE={rmse:.4f}, MAE={mae:.4f}\n")
        results.append({"pipeline": "B", "model": name, "RMSE": rmse, "MAE": mae})

        joblib.dump(model, f"models/{name}_pipelineB.pkl")
    except Exception as e:
        print(f"❌ {name} failed: {str(e)}\n")

print("\n=== PIPELINE A (Boosting Models) ===\n")
for name, model in pipeline_a_models.items():
    try:
        print(f"Training {name}...")
        model.fit(X_train_boost, y_train)
        preds = model.predict(X_val_boost)

        mse = mean_squared_error(y_val, preds)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_val, preds)

        print(f"{name} → RMSE={rmse:.4f}, MAE={mae:.4f}\n")
        results.append({"pipeline": "A", "model": name, "RMSE": rmse, "MAE": mae})

        joblib.dump(model, f"models/{name}_pipelineA.pkl")
    except Exception as e:
        print(f"❌ {name} failed: {str(e)}\n")

# ========================================
# Results & Best Model
# ========================================
results_df = pd.DataFrame(results).sort_values("RMSE")
print("\n=== MODEL COMPARISON ===\n")
print(results_df)

best_model_name = results_df.iloc[0]["model"]
best_pipeline = results_df.iloc[0]["pipeline"]

print(f"\n🏆 BEST MODEL: {best_model_name} (Pipeline {best_pipeline})")

best_model_file = f"models/{best_model_name}_pipeline{best_pipeline}.pkl"
best_model = joblib.load(best_model_file)

joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(FEATURE_COLS, "models/feature_columns_used.pkl")

results_df.to_csv("models/model_comparison.csv", index=False)

print("\n📌 Saved best model → models/best_model.pkl")
print("📌 Saved comparison → models/model_comparison.csv")
print("\n🎉 TRAINING COMPLETE!")
