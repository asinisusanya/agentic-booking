import pandas as pd
import numpy as np

def engineer_booking_features(bookings: pd.DataFrame) -> pd.DataFrame:
    df = bookings.copy()

    # --- Rename columns to match our pipeline expectations ---
    if "check_in_date" in df.columns:
        df.rename(columns={"check_in_date": "checkin"}, inplace=True)
    if "check_out_date" in df.columns:
        df.rename(columns={"check_out_date": "checkout"}, inplace=True)
    if "total_amount" in df.columns:
        df.rename(columns={"total_amount": "price_paid"}, inplace=True)
    if "num_guests" in df.columns:
        df.rename(columns={"num_guests": "number_of_guests"}, inplace=True)
    if "special_request" in df.columns:
        df.rename(columns={"special_request": "special_requests"}, inplace=True)

    # Ensure datetime
    for col in ["booking_date", "checkin", "checkout"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # --- Feature: stay nights ---
    if "nights_stay" in df.columns:
        df["stay_nights"] = df["nights_stay"].fillna(1)
    else:
        df["stay_nights"] = (df["checkout"] - df["checkin"]).dt.days.clip(lower=1)

    # --- Feature: lead time ---
    if "lead_time_days" not in df.columns:
        df["lead_time_days"] = (df["checkin"] - df["booking_date"]).dt.days.clip(lower=0)

    # --- Feature: price per night ---
    if "avg_daily_rate" in df.columns:
        df["price_per_night"] = df["avg_daily_rate"].fillna(df["price_paid"] / df["stay_nights"])
    else:
        df["price_per_night"] = df["price_paid"] / df["stay_nights"]

    # --- Weekend stay ---
    df["is_weekend_stay"] = df["checkin"].dt.weekday.isin([4, 5, 6]).astype(int)

    # --- Stay month ---
    df["checkin_month"] = df["checkin"].dt.month

    # --- Special requests count ---
    df["special_requests"] = df["special_requests"].fillna("")
    df["special_requests_count"] = df["special_requests"].apply(
        lambda x: len(x.split(";")) if isinstance(x, str) and x.strip() else 0
    )

    # --- Fill missing numeric values safely ---
    numeric_cols = [
        "stay_nights", "lead_time_days", "price_per_night",
        "is_weekend_stay", "number_of_guests", "special_requests_count"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


if __name__ == "__main__":
    print("Testing updated feature engineering...")
    # Load example without errors
    df_test = pd.DataFrame({
        "booking_date": ["2025-01-01"],
        "check_in_date": ["2025-01-10"],
        "check_out_date": ["2025-01-15"],
        "total_amount": [500],
        "avg_daily_rate": [100],
        "num_guests": [2],
        "special_request": ["Late checkin;High floor"]
    })

    df_test = engineer_booking_features(df_test)
    print(df_test)