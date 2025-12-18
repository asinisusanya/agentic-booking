import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/bookings_with_clusters.csv")

# -----------------------
# Cluster distribution
# -----------------------
plt.figure(figsize=(5,4))
df["cluster"].value_counts().plot(kind="bar")
plt.title("Distribution of Guest Clusters")
plt.xlabel("Cluster")
plt.ylabel("Number of Bookings")
plt.show()
