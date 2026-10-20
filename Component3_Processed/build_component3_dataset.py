import pandas as pd
import glob
import os
import re

# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATTERN = "CLEAN_House*.csv"
OUTPUT_DIR = "output"

RESAMPLE_INTERVAL = "15min"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# REFIT appliance mappings
# ------------------------------------------------------------

appliance_map = {
    1: [
        "Fridge", "Freezer", "Freezer", "Washer Dryer",
        "Washing Machine", "Dishwasher", "Computer",
        "Television", "Electric Heater"
    ],
    2: [
        "Fridge-Freezer", "Washing Machine", "Dishwasher",
        "Television", "Microwave", "Toaster", "Hi-Fi",
        "Kettle", "Overhead Fan"
    ],
    3: [
        "Toaster", "Fridge-Freezer", "Freezer", "Tumble Dryer",
        "Dishwasher", "Washing Machine", "Television",
        "Microwave", "Kettle"
    ],
    4: [
        "Fridge", "Freezer", "Fridge-Freezer",
        "Washing Machine", "Washing Machine",
        "Desktop Computer", "Television",
        "Microwave", "Kettle"
    ],
    5: [
        "Fridge-Freezer", "Tumble Dryer", "Washing Machine",
        "Dishwasher", "Desktop Computer",
        "Television", "Microwave", "Kettle", "Toaster"
    ],
    6: [
        "Freezer", "Washing Machine", "Dishwasher",
        "Computer", "Television", "Microwave",
        "Kettle", "Toaster", "Computer"
    ],
    7: [
        "Fridge", "Freezer", "Freezer", "Tumble Dryer",
        "Washing Machine", "Dishwasher",
        "Television", "Toaster", "Kettle"
    ],
    8: [
        "Fridge", "Freezer", "Washer Dryer",
        "Washing Machine", "Toaster",
        "Computer", "Television",
        "Microwave", "Kettle"
    ],
    9: [
        "Fridge-Freezer", "Washer Dryer", "Washing Machine",
        "Dishwasher", "Television",
        "Microwave", "Kettle",
        "Hi-Fi", "Electric Heater"
    ],
    10: [
        "Blender", "Toaster", "Chest Freezer",
        "Fridge-Freezer", "Washing Machine",
        "Dishwasher", "Television",
        "Microwave", "Food Mixer"
    ],
    11: [
        "Fridge", "Fridge-Freezer", "Washing Machine",
        "Dishwasher", "Computer", "Microwave",
        "Kettle", "Router", "Hi-Fi"
    ],
    12: [
        "Fridge-Freezer", "Unknown", "Unknown",
        "Computer", "Microwave", "Kettle",
        "Toaster", "Television", "Unknown"
    ],
    13: [
        "Television", "Freezer", "Washing Machine",
        "Dishwasher", "Unknown", "Network",
        "Microwave", "Microwave", "Kettle"
    ],
    15: [
        "Fridge-Freezer", "Tumble Dryer", "Washing Machine",
        "Dishwasher", "Computer", "Television",
        "Microwave", "Hi-Fi", "Toaster"
    ],
    16: [
        "Fridge-Freezer", "Fridge-Freezer",
        "Electric Heater", "Electric Heater",
        "Washing Machine", "Dishwasher",
        "Computer", "Television", "Dehumidifier"
    ],
    17: [
        "Freezer", "Fridge-Freezer", "Tumble Dryer",
        "Washing Machine", "Computer",
        "Television", "Microwave",
        "Kettle", "Television"
    ],
    18: [
        "Fridge", "Freezer", "Fridge-Freezer",
        "Washer Dryer", "Washing Machine",
        "Dishwasher", "Desktop Computer",
        "Television", "Microwave"
    ],
    19: [
        "Fridge-Freezer", "Washing Machine",
        "Television", "Microwave",
        "Kettle", "Toaster",
        "Bread Maker", "Games Console", "Hi-Fi"
    ],
    20: [
        "Fridge", "Freezer", "Tumble Dryer",
        "Washing Machine", "Dishwasher",
        "Computer", "Television",
        "Microwave", "Kettle"
    ],
    21: [
        "Fridge-Freezer", "Tumble Dryer",
        "Washing Machine", "Dishwasher",
        "Food Mixer", "Television",
        "Kettle", "Vivarium", "Pond Pump"
    ]
}


# ============================================================
# CATEGORY NORMALIZATION
# ============================================================

def normalize_category(name):
    name = name.lower().strip()

    if name in [
        "fridge",
        "fridge-freezer",
        "freezer",
        "chest freezer"
    ]:
        return "Refrigeration"

    if name == "washing machine":
        return "Washing Machine"

    if name == "dishwasher":
        return "Dishwasher"

    if name == "kettle":
        return "Kettle"

    return None


# ============================================================
# PROCESS ALL HOUSES
# ============================================================

files = glob.glob(CSV_PATTERN)

if not files:
    raise FileNotFoundError(
        "No CLEAN_House*.csv files were found in this folder."
    )

all_rows = []
house_summary = []

for file in sorted(files):

    match = re.search(r"House(\d+)", file)

    if not match:
        continue

    house_id = int(match.group(1))

    if house_id not in appliance_map:
        print(f"Skipping House {house_id}: no mapping.")
        continue

    print(f"\nProcessing House {house_id}...")

    # --------------------------------------------------------
    # Load required columns only
    # --------------------------------------------------------

    required_cols = [
        "Time",
        "Aggregate",
        "Issues",
        "Appliance1",
        "Appliance2",
        "Appliance3",
        "Appliance4",
        "Appliance5",
        "Appliance6",
        "Appliance7",
        "Appliance8",
        "Appliance9"
    ]

    df = pd.read_csv(
        file,
        usecols=lambda c: c in required_cols
    )

    original_rows = len(df)

    # --------------------------------------------------------
    # Datetime conversion
    # --------------------------------------------------------

    df["Time"] = pd.to_datetime(
        df["Time"],
        errors="coerce"
    )

    df = df.dropna(subset=["Time"])

    # --------------------------------------------------------
    # Remove REFIT issue rows
    # --------------------------------------------------------

    if "Issues" in df.columns:
        issue_rows = int((df["Issues"] == 1).sum())

        df = df[df["Issues"] == 0].copy()
    else:
        issue_rows = 0

    valid_rows = len(df)

    # --------------------------------------------------------
    # Ensure numeric values
    # --------------------------------------------------------

    df["Aggregate"] = pd.to_numeric(
        df["Aggregate"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Process each appliance channel
    # --------------------------------------------------------

    selected_channel_count = 0

    for appliance_number in range(1, 10):

        col = f"Appliance{appliance_number}"

        if col not in df.columns:
            continue

        raw_appliance_name = appliance_map[house_id][appliance_number - 1]

        category = normalize_category(raw_appliance_name)

        if category is None:
            continue

        selected_channel_count += 1

        temp = df[
            ["Time", "Aggregate", col]
        ].copy()

        temp[col] = pd.to_numeric(
            temp[col],
            errors="coerce"
        )

        temp = temp.dropna(
            subset=["Aggregate", col]
        )

        # ----------------------------------------------------
        # Set timestamp as index for resampling
        # ----------------------------------------------------

        temp = temp.set_index("Time")

        # ----------------------------------------------------
        # Resample to 15 minutes
        # ----------------------------------------------------

        resampled = temp.resample(
            RESAMPLE_INTERVAL
        ).agg({
            "Aggregate": ["mean", "max"],
            col: ["mean", "max"]
        })

        resampled.columns = [
            "aggregate_mean_w",
            "aggregate_max_w",
            "appliance_mean_w",
            "appliance_max_w"
        ]

        resampled = resampled.reset_index()

        # Remove intervals with no usable observations
        resampled = resampled.dropna(
            subset=[
                "aggregate_mean_w",
                "appliance_mean_w"
            ]
        )

        # ----------------------------------------------------
        # Activity status
        # ----------------------------------------------------

        # Initial threshold:
        # more than 10 W average inside 15-minute window
        resampled["appliance_active"] = (
            resampled["appliance_mean_w"] > 10
        ).astype(int)

        resampled["house_id"] = house_id
        resampled["channel"] = col

        resampled["original_appliance_name"] = (
            raw_appliance_name
        )

        resampled["appliance_category"] = category

        # ----------------------------------------------------
        # Time features
        # ----------------------------------------------------

        resampled["hour"] = resampled["Time"].dt.hour
        resampled["day_of_week"] = resampled["Time"].dt.dayofweek
        resampled["is_weekend"] = (
            resampled["day_of_week"] >= 5
        ).astype(int)

        # ----------------------------------------------------
        # Reorder columns
        # ----------------------------------------------------

        resampled = resampled[
            [
                "Time",
                "house_id",
                "channel",
                "original_appliance_name",
                "appliance_category",
                "aggregate_mean_w",
                "aggregate_max_w",
                "appliance_mean_w",
                "appliance_max_w",
                "appliance_active",
                "hour",
                "day_of_week",
                "is_weekend"
            ]
        ]

        all_rows.append(resampled)

    house_summary.append({
        "house_id": house_id,
        "original_rows": original_rows,
        "valid_rows_after_issue_removal": valid_rows,
        "removed_issue_rows": issue_rows,
        "selected_appliance_channels": selected_channel_count
    })


# ============================================================
# COMBINE DATA
# ============================================================

if not all_rows:
    raise RuntimeError(
        "No selected appliance records were generated."
    )

final_df = pd.concat(
    all_rows,
    ignore_index=True
)

# ------------------------------------------------------------
# Basic cleaning
# ------------------------------------------------------------

final_df = final_df.drop_duplicates()

final_df = final_df.sort_values(
    ["house_id", "Time", "appliance_category"]
)

# Prevent impossible negative power values
power_cols = [
    "aggregate_mean_w",
    "aggregate_max_w",
    "appliance_mean_w",
    "appliance_max_w"
]

for col in power_cols:
    final_df = final_df[
        final_df[col] >= 0
    ]


# ============================================================
# SAVE MAIN DATASET
# ============================================================

main_output = os.path.join(
    OUTPUT_DIR,
    "component3_refit_15min.csv"
)

final_df.to_csv(
    main_output,
    index=False
)


# ============================================================
# SAVE HOUSE PROCESSING SUMMARY
# ============================================================

house_summary_df = pd.DataFrame(
    house_summary
)

house_output = os.path.join(
    OUTPUT_DIR,
    "component3_house_processing_summary.csv"
)

house_summary_df.to_csv(
    house_output,
    index=False
)


# ============================================================
# SAVE CATEGORY SUMMARY
# ============================================================

category_summary = (
    final_df
    .groupby("appliance_category")
    .agg(
        houses=("house_id", "nunique"),
        records=("appliance_category", "size"),
        active_records=("appliance_active", "sum"),
        average_power_w=("appliance_mean_w", "mean"),
        max_power_w=("appliance_max_w", "max")
    )
    .reset_index()
)

category_summary["active_percent"] = (
    category_summary["active_records"]
    / category_summary["records"]
    * 100
)

category_output = os.path.join(
    OUTPUT_DIR,
    "component3_category_summary_15min.csv"
)

category_summary.to_csv(
    category_output,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n======================================")
print("COMPONENT 3 DATASET BUILD COMPLETED")
print("======================================")

print(f"\nTotal final records: {len(final_df):,}")

print(
    f"Unique houses: "
    f"{final_df['house_id'].nunique()}"
)

print("\nAppliance category counts:")

print(
    final_df["appliance_category"]
    .value_counts()
)

print("\nCategory summary:")

print(
    category_summary.to_string(index=False)
)

print("\nGenerated files:")

print(main_output)
print(house_output)
print(category_output)