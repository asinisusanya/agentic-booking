import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib

DATA_PATH = "data/bookings_with_clusters.csv"

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

print("📂 Loading dataset with cluster labels...")
df = pd.read_csv(DATA_PATH)

# --- Prepare data ---
X = df[cluster_features].copy()
X = X.fillna(X.median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Load trained KMeans model ---
km = joblib.load("models/kmeans_model.pkl")

# --- Apply PCA for visualization ---
print("🔍 Running PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df["pc1"] = X_pca[:, 0]
df["pc2"] = X_pca[:, 1]

# --- Plot clusters ---
print("📊 Plotting clusters...")
plt.figure(figsize=(10, 6))

colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'olive']

for cluster_id in sorted(df["cluster"].unique()):
    subset = df[df["cluster"] == cluster_id]
    plt.scatter(subset["pc1"], subset["pc2"], 
                s=40, 
                alpha=0.7,
                color=colors[int(cluster_id) % len(colors)],
                label=f"Cluster {cluster_id}")

plt.title("Customer Clusters (PCA 2D Projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)

plt.savefig("data/cluster_visualization.png", dpi=300)
plt.show()

print("📁 Cluster plot saved → data/cluster_visualization.png")
