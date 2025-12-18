import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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

# Load data
df = pd.read_csv(DATA_PATH)

# Prepare feature matrix
X = df[cluster_features].fillna(df[cluster_features].median())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia_list = []
silhouette_list = []
K_range = range(2, 13)

for k in K_range:
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(X_scaled)
    
    inertia_list.append(km.inertia_)
    silhouette_list.append(silhouette_score(X_scaled, labels))


# ----- Elbow Plot -----
plt.figure(figsize=(8, 4))
plt.plot(K_range, inertia_list, marker='o')
plt.title("Elbow Method (Inertia vs K)")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.grid(True)
plt.savefig("data/elbow_plot.png", dpi=300)
plt.show(block=True)

# ----- Silhouette Plot -----
plt.figure(figsize=(8, 4))
plt.plot(K_range, silhouette_list, marker='o', color='green')
plt.title("Silhouette Score vs K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.savefig("data/silhouette_plot.png", dpi=300)
plt.show(block=True)
