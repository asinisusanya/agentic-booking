import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/feature_bookings.csv")

# -----------------------
# Rating distribution
# -----------------------
plt.figure(figsize=(6,4))
sns.histplot(df["rating"].dropna(), bins=20, kde=True)
plt.title("Distribution of Guest Ratings")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

# -----------------------
# Lead time vs rating
# -----------------------
plt.figure(figsize=(6,4))
sns.scatterplot(x="lead_time_days", y="rating", data=df, alpha=0.3)
plt.title("Lead Time vs Rating")
plt.show()

# -----------------------
# Price per night vs rating
# -----------------------
plt.figure(figsize=(6,4))
sns.scatterplot(x="price_per_night", y="rating", data=df, alpha=0.3)
plt.title("Price per Night vs Rating")
plt.show()
