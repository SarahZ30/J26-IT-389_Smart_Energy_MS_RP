import pandas as pd
import os

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "output/component3_feature_fusion_dataset.csv"
OUTPUT_DIR = "output"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "component3_labeled_dataset.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "component3_label_summary.csv"
)

MATRIX_FILE = os.path.join(
    OUTPUT_DIR,
    "component3_appliance_action_matrix.csv"
)

SAMPLE_FILE = os.path.join(
    OUTPUT_DIR,
    "component3_labeled_sample.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# 1. LOAD FEATURE-FUSION DATASET
# ============================================================

print("========================================")
print("COMPONENT 3 - SCHEDULING LABEL CREATION")
print("========================================")

print("\nLoading feature-fusion dataset...")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["Time"]
)

print(f"Records loaded: {len(df):,}")


# ============================================================
# 2. BASIC VALIDATION
# ============================================================

required_columns = [
    "appliance_category",
    "appliance_active",
    "occupancy",
    "critical_flag",
    "flexible_flag",
    "high_power_short_use",
    "tariff_band",
    "predicted_load_level",
    "user_preference",
    "appliance_load_ratio"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("Required columns validated successfully.")


# ============================================================
# 3. SCHEDULING POLICY
# ============================================================

def generate_scheduling_action(row):

    appliance = row["appliance_category"]

    active = int(row["appliance_active"])
    occupancy = int(row["occupancy"])

    critical = int(row["critical_flag"])
    flexible = int(row["flexible_flag"])
    high_power_short = int(row["high_power_short_use"])

    tariff = row["tariff_band"]
    load_level = row["predicted_load_level"]

    preference = int(row["user_preference"])

    load_ratio = float(row["appliance_load_ratio"])


    # ========================================================
    # RULE 1
    # Inactive appliances remain OFF.
    # ========================================================

    if active == 0:

        return (
            "TURN_OFF",
            "Appliance is currently inactive"
        )


    # ========================================================
    # RULE 2
    # Active critical refrigeration receives priority.
    #
    # Important:
    # Only ACTIVE refrigeration is assigned PRIORITY.
    # Inactive refrigeration records were already handled
    # by Rule 1.
    # ========================================================

    if (
        critical == 1
        and active == 1
    ):

        return (
            "PRIORITY",
            "Active critical refrigeration load receives operational priority"
        )


    # ========================================================
    # RULE 3
    # Flexible appliances may be delayed when:
    #
    # - appliance is active
    # - appliance is flexible
    # - tariff is Peak
    # - predicted household demand is High
    # - user allows at least some flexibility
    #
    # Typical examples:
    # Washing Machine
    # Dishwasher
    # ========================================================

    if (
        active == 1
        and flexible == 1
        and tariff == "Peak"
        and load_level == "High"
        and preference >= 1
    ):

        return (
            "DELAY",
            "Active flexible appliance delayed during peak tariff and high predicted demand"
        )


    # ========================================================
    # RULE 4
    # Load balancing.
    #
    # Applied when:
    # - appliance is active
    # - predicted household load is High
    # - appliance contributes significantly to current load
    #   OR it is categorized as a short-use high-power load
    #
    # Kettle is especially suitable here.
    # ========================================================

    if (
        active == 1
        and load_level == "High"
        and (
            load_ratio >= 0.25
            or high_power_short == 1
        )
    ):

        return (
            "LOAD_BALANCE",
            "Active appliance contributes significantly during high predicted demand"
        )


    # ========================================================
    # RULE 5
    # Eco mode.
    #
    # Active non-critical appliances may operate under a
    # reduced-energy policy when:
    #
    # - predicted load is Medium
    # OR
    # - tariff is Peak
    #
    # This is a scheduling-policy abstraction for the PP1 PoC.
    # ========================================================

    if (
        active == 1
        and critical == 0
        and (
            load_level == "Medium"
            or tariff == "Peak"
        )
    ):

        return (
            "ECO_MODE",
            "Active non-critical appliance operated under reduced-energy policy"
        )


    # ========================================================
    # RULE 6
    # Normal ON operation while occupied.
    # ========================================================

    if (
        active == 1
        and occupancy == 1
    ):

        return (
            "TURN_ON",
            "Active appliance permitted normal operation while household is occupied"
        )


    # ========================================================
    # RULE 7
    # Default active behavior.
    #
    # If none of the above restrictions apply,
    # allow normal operation.
    # ========================================================

    return (
        "TURN_ON",
        "Normal appliance operation permitted"
    )


# ============================================================
# 4. APPLY SCHEDULING POLICY
# ============================================================

print("\nGenerating scheduling labels...")

results = df.apply(
    generate_scheduling_action,
    axis=1
)

df["target_action"] = [
    result[0]
    for result in results
]

df["label_reason"] = [
    result[1]
    for result in results
]


# ============================================================
# 5. TARGET ACTION IDs
# ============================================================

action_map = {
    "TURN_ON": 0,
    "TURN_OFF": 1,
    "DELAY": 2,
    "ECO_MODE": 3,
    "PRIORITY": 4,
    "LOAD_BALANCE": 5
}

df["target_action_id"] = (
    df["target_action"]
    .map(action_map)
)


# ============================================================
# 6. VALIDATE GENERATED LABELS
# ============================================================

if df["target_action"].isna().any():
    raise ValueError(
        "Some rows do not have target_action labels."
    )

if df["target_action_id"].isna().any():
    raise ValueError(
        "Some scheduling actions could not be mapped to IDs."
    )

df["target_action_id"] = (
    df["target_action_id"]
    .astype(int)
)

expected_actions = {
    "TURN_ON",
    "TURN_OFF",
    "DELAY",
    "ECO_MODE",
    "PRIORITY",
    "LOAD_BALANCE"
}

generated_actions = set(
    df["target_action"].unique()
)

missing_actions = (
    expected_actions - generated_actions
)

if missing_actions:
    print(
        "\nWARNING:"
        f" These action classes were not generated: "
        f"{missing_actions}"
    )


# ============================================================
# 7. LOGICAL VALIDATION CHECKS
# ============================================================

print("\nRunning logical validation checks...")


# ------------------------------------------------------------
# Check 1:
# DELAY must never occur when appliance is inactive.
# ------------------------------------------------------------

invalid_delay = df[
    (df["target_action"] == "DELAY")
    &
    (df["appliance_active"] == 0)
]

if len(invalid_delay) > 0:
    print(
        f"WARNING: Found {len(invalid_delay)} inactive "
        f"appliances labeled DELAY."
    )
else:
    print(
        "PASS: No inactive appliance is labeled DELAY."
    )


# ------------------------------------------------------------
# Check 2:
# PRIORITY should only apply to active critical appliances.
# ------------------------------------------------------------

invalid_priority = df[
    (df["target_action"] == "PRIORITY")
    &
    (
        (df["critical_flag"] != 1)
        |
        (df["appliance_active"] != 1)
    )
]

if len(invalid_priority) > 0:
    print(
        f"WARNING: Found {len(invalid_priority)} invalid "
        f"PRIORITY records."
    )
else:
    print(
        "PASS: PRIORITY only applies to active critical loads."
    )


# ------------------------------------------------------------
# Check 3:
# Inactive appliances should be TURN_OFF.
# ------------------------------------------------------------

invalid_inactive = df[
    (df["appliance_active"] == 0)
    &
    (df["target_action"] != "TURN_OFF")
]

if len(invalid_inactive) > 0:
    print(
        f"WARNING: Found {len(invalid_inactive)} inactive "
        f"appliances not labeled TURN_OFF."
    )
else:
    print(
        "PASS: All inactive appliances are TURN_OFF."
    )


# ============================================================
# 8. SAVE FULL LABELED DATASET
# ============================================================

print("\nSaving full labeled dataset...")

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 9. CREATE LABEL SUMMARY
# ============================================================

label_summary = (
    df
    .groupby("target_action")
    .agg(

        records=(
            "target_action",
            "size"
        ),

        houses=(
            "house_id",
            "nunique"
        ),

        appliances=(
            "appliance_category",
            "nunique"
        ),

        avg_predicted_load=(
            "predicted_load_w",
            "mean"
        ),

        avg_appliance_power=(
            "appliance_mean_w",
            "mean"
        ),

        occupancy_percent=(
            "occupancy",
            "mean"
        )

    )
    .reset_index()
)

label_summary["percentage"] = (
    label_summary["records"]
    / len(df)
    * 100
)

label_summary["occupancy_percent"] = (
    label_summary["occupancy_percent"]
    * 100
)

label_summary = label_summary.sort_values(
    "records",
    ascending=False
).reset_index(drop=True)

label_summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# 10. CREATE APPLIANCE x ACTION MATRIX
# ============================================================

matrix = pd.crosstab(
    df["appliance_category"],
    df["target_action"]
)

matrix.to_csv(
    MATRIX_FILE
)


# ============================================================
# 11. CREATE SMALL BALANCED SAMPLE
# ============================================================

sample_parts = []

for action, group in df.groupby(
    "target_action"
):

    sample_size = min(
        1000,
        len(group)
    )

    sampled = group.sample(
        n=sample_size,
        random_state=RANDOM_STATE
    )

    sample_parts.append(
        sampled
    )

sample_df = pd.concat(
    sample_parts,
    ignore_index=True
)

# Shuffle final sample.
sample_df = sample_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

sample_df.to_csv(
    SAMPLE_FILE,
    index=False
)


# ============================================================
# 12. CREATE REASON SUMMARY
# ============================================================

reason_summary = (
    df
    .groupby(
        [
            "target_action",
            "label_reason"
        ]
    )
    .size()
    .reset_index(
        name="records"
    )
)

reason_summary_file = os.path.join(
    OUTPUT_DIR,
    "component3_label_reason_summary.csv"
)

reason_summary.to_csv(
    reason_summary_file,
    index=False
)


# ============================================================
# 13. FINAL TERMINAL OUTPUT
# ============================================================

print("\n========================================")
print("LABEL GENERATION COMPLETED")
print("========================================")

print(
    f"\nTotal records: "
    f"{len(df):,}"
)

print("\nTarget action counts:")

print(
    df["target_action"]
    .value_counts()
)


print("\nTarget action percentages:")

action_percentages = (
    df["target_action"]
    .value_counts(
        normalize=True
    )
    * 100
)

print(
    action_percentages
    .round(2)
)


print("\nAppliance x Action matrix:")

print(
    matrix
)


print("\nLabel summary:")

print(
    label_summary
    .to_string(
        index=False
    )
)


print("\nLabel reason summary:")

print(
    reason_summary
    .to_string(
        index=False
    )
)


# ============================================================
# 14. EXTRA CHECK:
# DISPLAY ACTIONS FOR EACH APPLIANCE
# ============================================================

print("\nActions generated for each appliance:")

for appliance in sorted(
    df["appliance_category"]
    .unique()
):

    actions = (
        df[
            df["appliance_category"]
            == appliance
        ]["target_action"]
        .value_counts()
    )

    print(
        f"\n{appliance}:"
    )

    print(actions)


# ============================================================
# 15. GENERATED FILE LIST
# ============================================================

print("\nGenerated files:")

print(
    f"1. {OUTPUT_FILE}"
)

print(
    f"2. {SUMMARY_FILE}"
)

print(
    f"3. {MATRIX_FILE}"
)

print(
    f"4. {SAMPLE_FILE}"
)

print(
    f"5. {reason_summary_file}"
)

print("\nDone.")