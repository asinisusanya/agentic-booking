import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import joblib


# 1. Load processed dataset
DATA_PATH = "data/feature_bookings.csv"

cluster_features = [
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


def load_dataset():
    df = pd.read_csv(DATA_PATH)
    return df


def prepare_clustering_data(df):
    # Select features and handle missing values
    X = df[cluster_features].copy()

    for col in cluster_features:
        X[col] = X[col].fillna(X[col].median())

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


def evaluate_k_range(X_scaled, k_min=2, k_max=12):
    inertias = []
    silhouettes = []

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(X_scaled)

        inertia = km.inertia_
        sil = silhouette_score(X_scaled, labels)

        inertias.append(inertia)
        silhouettes.append(sil)

        print(f"K={k} → inertia={inertia:.2f}, silhouette={sil:.4f}")

    return inertias, silhouettes


def choose_best_k(silhouettes):
    # Choose the k with the highest silhouette
    best_k = np.argmax(silhouettes) + 2  # +2 because range starts at k=2
    return best_k


def train_final_kmeans(X_scaled, k):
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    km.fit(X_scaled)
    return km


def save_clustered_dataset(df, labels):
    df["cluster"] = labels
    df.to_csv("data/bookings_with_clusters.csv", index=False)
    print("✓ Saved clustering results → data/bookings_with_clusters.csv")


if __name__ == "__main__":
    print("🔄 Loading dataset...")
    df = load_dataset()

    print("📦 Preparing clustering matrix...")
    X_scaled, scaler = prepare_clustering_data(df)

    print("🔍 Evaluating K from 2 to 12...")
    inertias, silhouettes = evaluate_k_range(X_scaled)

    best_k = choose_best_k(silhouettes)
    print(f"\n⭐ Best K selected = {best_k}\n")

    print("🏗 Training final K-means model...")
    km = train_final_kmeans(X_scaled, best_k)

    print("💾 Saving scaler and KMeans model...")
    joblib.dump(scaler, "models/kmeans_scaler.pkl")
    joblib.dump(km, "models/kmeans_model.pkl")

    print("🏷 Assigning clusters to full dataset...")
    labels = km.predict(X_scaled)
    save_clustered_dataset(df, labels)

    print("🎉 Clustering completed successfully!")
