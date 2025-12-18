import pandas as pd
from features import engineer_booking_features

# File paths
BOOKINGS_PATH = "data/cinnamon_bookings.csv"
FEEDBACK_PATH = "data/cinnamon_feedback.csv"
PROFILES_PATH = "data/customers_profiles.csv"


def load_raw_data():
    """Load files with safe encoding."""
    try:
        bookings = pd.read_csv(BOOKINGS_PATH, encoding="utf-8")
    except:
        bookings = pd.read_csv(BOOKINGS_PATH, encoding="latin1")

    try:
        feedback = pd.read_csv(FEEDBACK_PATH, encoding="utf-8")
    except:
        feedback = pd.read_csv(FEEDBACK_PATH, encoding="latin1")

    try:
        profiles = pd.read_csv(PROFILES_PATH, encoding="utf-8")
    except:
        profiles = pd.read_csv(PROFILES_PATH, encoding="latin1")

    return bookings, feedback, profiles


def preprocess_bookings(bookings):
    """Convert to datetime + feature engineering."""
    date_cols = ["booking_date", "check_in_date", "check_out_date"]
    for col in date_cols:
        if col in bookings.columns:
            bookings[col] = pd.to_datetime(bookings[col], errors="coerce")

    # Engineer ML features
    return engineer_booking_features(bookings)


def merge_feedback(bookings, feedback):
    """
    Merge based on customer_id.
    Feedback contains multiple rating columns; we focus on overall_rating.
    """
    if "customer_id" in bookings.columns and "customer_id" in feedback.columns:
        merged = bookings.merge(feedback, on="customer_id", how="left")
        print("✓ Merged feedback using customer_id")
    else:
        print("⚠ Cannot merge feedback: customer_id missing.")
        return bookings

    # Rename overall_rating → rating
    if "overall_rating" in merged.columns:
        merged.rename(columns={"overall_rating": "rating"}, inplace=True)

    return merged


def merge_profiles(data, profiles):
    """Merge profiles on customer_id."""
    if "customer_id" in data.columns and "customer_id" in profiles.columns:
        merged = data.merge(profiles, on="customer_id", how="left")
        print("✓ Merged profiles using customer_id")
        return merged

    print("⚠ Cannot merge profiles: customer_id missing.")
    return data


if __name__ == "__main__":
    print("🔄 Loading raw data...")
    bookings, feedback, profiles = load_raw_data()

    print("🔧 Preprocessing bookings...")
    bookings_processed = preprocess_bookings(bookings)

    print("🔗 Merging feedback...")
    merged1 = merge_feedback(bookings_processed, feedback)

    print("👤 Merging profiles...")
    final_data = merge_profiles(merged1, profiles)

    print("💾 Saving → data/feature_bookings.csv")
    final_data.to_csv("data/feature_bookings.csv", index=False)

    print("🎉 DONE — Your ML dataset is fully prepared!")
