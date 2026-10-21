import pandas as pd
import glob
import os
import re

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

CSV_PATTERN = "CLEAN_House*.csv"

# Appliance mappings from the official REFIT README.
# Only appliance names are required for this research audit.

appliance_map = {
    1: [
        "Fridge",
        "Freezer",
        "Freezer",
        "Washer Dryer",
        "Washing Machine",
        "Dishwasher",
        "Computer",
        "Television",
        "Electric Heater"
    ],

    2: [
        "Fridge-Freezer",
        "Washing Machine",
        "Dishwasher",
        "Television",
        "Microwave",
        "Toaster",
        "Hi-Fi",
        "Kettle",
        "Overhead Fan"
    ],

    3: [
        "Toaster",
        "Fridge-Freezer",
        "Freezer",
        "Tumble Dryer",
        "Dishwasher",
        "Washing Machine",
        "Television",
        "Microwave",
        "Kettle"
    ],

    4: [
        "Fridge",
        "Freezer",
        "Fridge-Freezer",
        "Washing Machine",
        "Washing Machine",
        "Desktop Computer",
        "Television",
        "Microwave",
        "Kettle"
    ],

    5: [
        "Fridge-Freezer",
        "Tumble Dryer",
        "Washing Machine",
        "Dishwasher",
        "Desktop Computer",
        "Television",
        "Microwave",
        "Kettle",
        "Toaster"
    ],

    6: [
        "Freezer",
        "Washing Machine",
        "Dishwasher",
        "Computer",
        "Television",
        "Microwave",
        "Kettle",
        "Toaster",
        "Computer"
    ],

    7: [
        "Fridge",
        "Freezer",
        "Freezer",
        "Tumble Dryer",
        "Washing Machine",
        "Dishwasher",
        "Television",
        "Toaster",
        "Kettle"
    ],

    8: [
        "Fridge",
        "Freezer",
        "Washer Dryer",
        "Washing Machine",
        "Toaster",
        "Computer",
        "Television",
        "Microwave",
        "Kettle"
    ],

    9: [
        "Fridge-Freezer",
        "Washer Dryer",
        "Washing Machine",
        "Dishwasher",
        "Television",
        "Microwave",
        "Kettle",
        "Hi-Fi",
        "Electric Heater"
    ],

    10: [
        "Blender",
        "Toaster",
        "Chest Freezer",
        "Fridge-Freezer",
        "Washing Machine",
        "Dishwasher",
        "Television",
        "Microwave",
        "Food Mixer"
    ],

    11: [
        "Fridge",
        "Fridge-Freezer",
        "Washing Machine",
        "Dishwasher",
        "Computer",
        "Microwave",
        "Kettle",
        "Router",
        "Hi-Fi"
    ],

    12: [
        "Fridge-Freezer",
        "Unknown",
        "Unknown",
        "Computer",
        "Microwave",
        "Kettle",
        "Toaster",
        "Television",
        "Unknown"
    ],

    13: [
        "Television",
        "Freezer",
        "Washing Machine",
        "Dishwasher",
        "Unknown",
        "Network",
        "Microwave",
        "Microwave",
        "Kettle"
    ],

    15: [
        "Fridge-Freezer",
        "Tumble Dryer",
        "Washing Machine",
        "Dishwasher",
        "Computer",
        "Television",
        "Microwave",
        "Hi-Fi",
        "Toaster"
    ],

    16: [
        "Fridge-Freezer",
        "Fridge-Freezer",
        "Electric Heater",
        "Electric Heater",
        "Washing Machine",
        "Dishwasher",
        "Computer",
        "Television",
        "Dehumidifier"
    ],

    17: [
        "Freezer",
        "Fridge-Freezer",
        "Tumble Dryer",
        "Washing Machine",
        "Computer",
        "Television",
        "Microwave",
        "Kettle",
        "Television"
    ],

    18: [
        "Fridge",
        "Freezer",
        "Fridge-Freezer",
        "Washer Dryer",
        "Washing Machine",
        "Dishwasher",
        "Desktop Computer",
        "Television",
        "Microwave"
    ],

    19: [
        "Fridge-Freezer",
        "Washing Machine",
        "Television",
        "Microwave",
        "Kettle",
        "Toaster",
        "Bread Maker",
        "Games Console",
        "Hi-Fi"
    ],

    20: [
        "Fridge",
        "Freezer",
        "Tumble Dryer",
        "Washing Machine",
        "Dishwasher",
        "Computer",
        "Television",
        "Microwave",
        "Kettle"
    ],

    21: [
        "Fridge-Freezer",
        "Tumble Dryer",
        "Washing Machine",
        "Dishwasher",
        "Food Mixer",
        "Television",
        "Kettle",
        "Vivarium",
        "Pond Pump"
    ]
}


# ---------------------------------------------------------
# PROCESS FILES
# ---------------------------------------------------------

files = glob.glob(CSV_PATTERN)

if not files:
    print("No CLEAN_House*.csv files found.")
    exit()

results = []

for file in files:

    match = re.search(r"House(\d+)", file)

    if not match:
        continue

    house = int(match.group(1))

    print(f"Processing House {house}...")

    if house not in appliance_map:
        print(f"No appliance mapping for House {house}. Skipping.")
        continue

    df = pd.read_csv(file)

    total_rows = len(df)

    # Remove rows already marked problematic by REFIT
    if "Issues" in df.columns:
        valid_df = df[df["Issues"] == 0].copy()
    else:
        valid_df = df.copy()

    valid_rows = len(valid_df)

    for appliance_num in range(1, 10):

        col = f"Appliance{appliance_num}"

        if col not in valid_df.columns:
            continue

        appliance_name = appliance_map[house][appliance_num - 1]

        values = pd.to_numeric(valid_df[col], errors="coerce")

        nonzero = values[values > 0]

        nonzero_count = len(nonzero)

        active_percent = (
            nonzero_count / valid_rows * 100
            if valid_rows > 0
            else 0
        )

        mean_all = values.mean()

        mean_active = nonzero.mean() if nonzero_count > 0 else 0

        max_watts = values.max()

        results.append({
            "House": house,
            "Channel": col,
            "Appliance_Name": appliance_name,
            "Total_Rows": total_rows,
            "Valid_Rows": valid_rows,
            "NonZero_Count": nonzero_count,
            "Active_Percent": round(active_percent, 4),
            "Mean_Watts_All": round(mean_all, 4),
            "Mean_Watts_Active": round(mean_active, 4),
            "Max_Watts": round(max_watts, 4)
        })


# ---------------------------------------------------------
# SAVE DETAILED SUMMARY
# ---------------------------------------------------------

detail_df = pd.DataFrame(results)

detail_df = detail_df.sort_values(
    ["Appliance_Name", "House", "Channel"]
)

detail_df.to_csv(
    "refit_appliance_category_summary.csv",
    index=False
)


# ---------------------------------------------------------
# CATEGORY-LEVEL SUMMARY
# ---------------------------------------------------------

category_df = (
    detail_df
    .groupby("Appliance_Name")
    .agg(
        Houses_Available=("House", "nunique"),
        Channels=("Channel", "count"),
        Total_NonZero_Readings=("NonZero_Count", "sum"),
        Avg_Active_Percent=("Active_Percent", "mean"),
        Avg_Active_Watts=("Mean_Watts_Active", "mean"),
        Maximum_Watts=("Max_Watts", "max")
    )
    .reset_index()
)

category_df = category_df.sort_values(
    "Houses_Available",
    ascending=False
)

category_df.to_csv(
    "refit_appliance_overall_summary.csv",
    index=False
)


print("\nCompleted successfully.")

print("\nCreated files:")
print("1. refit_appliance_category_summary.csv")
print("2. refit_appliance_overall_summary.csv")

print("\nTop appliance categories:")
print(category_df.head(15).to_string(index=False))