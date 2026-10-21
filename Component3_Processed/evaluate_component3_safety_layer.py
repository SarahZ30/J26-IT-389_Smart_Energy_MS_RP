import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "v2_output/component3_v2_dataset.csv"

MODEL_FILE = (
    "ablation_final_output/models/"
    "E_Personalization_XGBoost.joblib"
)

LABEL_ENCODER_FILE = (
    "ablation_final_output/models/"
    "target_label_encoder.joblib"
)

OUTPUT_DIR = "safety_layer_output"

RESULT_DIR = os.path.join(
    OUTPUT_DIR,
    "results"
)

PLOT_DIR = os.path.join(
    OUTPUT_DIR,
    "plots"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

os.makedirs(
    PLOT_DIR,
    exist_ok=True
)

RANDOM_STATE = 42

rng = np.random.default_rng(
    RANDOM_STATE
)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("============================================")
print("COMPONENT 3 - SAFETY LAYER EVALUATION")
print("============================================")

print("\nLoading V2 dataset...")

df = pd.read_csv(
    DATA_FILE
)

print(
    f"Total records: {len(df):,}"
)

print(
    f"Unique houses: "
    f"{df['house_id'].nunique()}"
)


# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================

print("\nLoading selected XGBoost model...")

model = joblib.load(
    MODEL_FILE
)

label_encoder = joblib.load(
    LABEL_ENCODER_FILE
)

print(
    "Model loaded successfully."
)

print(
    "Selected model: "
    "XGBoost + Personalization"
)


# ============================================================
# 3. RECREATE ORIGINAL TEST-HOUSE SPLIT
# ============================================================

groups = df[
    "house_id"
]

y_dummy = df[
    "target_action"
]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_val_idx, test_idx = next(
    splitter.split(
        df,
        y_dummy,
        groups=groups
    )
)

test_df = (
    df.iloc[
        test_idx
    ]
    .copy()
    .reset_index(drop=True)
)

print("\nTest-set information:")

print(
    f"Test records: "
    f"{len(test_df):,}"
)

print(
    "Test houses:"
)

print(
    sorted(
        test_df[
            "house_id"
        ]
        .unique()
        .tolist()
    )
)


# ============================================================
# 4. GENERATE ML RECOMMENDATIONS
# ============================================================

print(
    "\nGenerating XGBoost scheduling recommendations..."
)

predicted_ids = model.predict(
    test_df
)

test_df[
    "ml_action"
] = label_encoder.inverse_transform(
    predicted_ids.astype(int)
)

print(
    "ML recommendations generated."
)


# ============================================================
# 5. SAFETY RULE IDENTIFICATION
# ============================================================

def identify_safety_violation(
    row,
    action
):

    # Ensure action is always treated as a scalar string.
    action = str(action)

    appliance = str(
        row["appliance_category"]
    )

    active = int(
        row["appliance_active"]
    )

    interruptible = int(
        row["interruptible"]
    )

    off_duration = float(
        row["off_duration_minutes"]
    )

    max_off = max(
        float(
            row["max_off_minutes"]
        ),
        1.0
    )

    max_delay = max(
        float(
            row["max_delay_minutes"]
        ),
        1.0
    )


    # ========================================================
    # RULE S1
    # Critical refrigeration has exceeded maximum OFF period.
    # ========================================================

    if (
        appliance == "Refrigeration"
        and off_duration >= max_off
        and action not in [
            "TURN_ON",
            "PRIORITY"
        ]
    ):

        return (
            True,
            "S1_CRITICAL_MAX_OFF_EXCEEDED"
        )


    # ========================================================
    # RULE S2
    # Refrigeration should not receive DELAY.
    # ========================================================

    if (
        appliance == "Refrigeration"
        and action == "DELAY"
    ):

        return (
            True,
            "S2_CRITICAL_APPLIANCE_DELAY"
        )


    # ========================================================
    # RULE S3
    # Running non-interruptible appliances must not be
    # interrupted or delayed while their cycle is active.
    # ========================================================

    if (
        appliance in [
            "Washing Machine",
            "Dishwasher"
        ]
        and active == 1
        and interruptible == 0
        and action in [
            "TURN_OFF",
            "DELAY"
        ]
    ):

        return (
            True,
            "S3_NON_INTERRUPTIBLE_CYCLE"
        )


    # ========================================================
    # RULE S4
    # Flexible appliance cannot remain delayed beyond the
    # experimental configured maximum delay.
    # ========================================================

    if (
        appliance in [
            "Washing Machine",
            "Dishwasher"
        ]
        and active == 0
        and off_duration >= max_delay
        and action == "DELAY"
    ):

        return (
            True,
            "S4_MAX_DELAY_EXCEEDED"
        )


    return (
        False,
        "SAFE"
    )


# ============================================================
# 6. SAFETY OVERRIDE LOGIC
# ============================================================

def apply_safety_layer(
    row,
    ml_action
):

    ml_action = str(
        ml_action
    )

    violation, rule = (
        identify_safety_violation(
            row,
            ml_action
        )
    )


    # --------------------------------------------------------
    # No violation
    # --------------------------------------------------------

    if not violation:

        return pd.Series({

            "final_action":
                ml_action,

            "safety_override":
                0,

            "safety_rule":
                "NONE",

            "original_violation":
                0
        })


    # --------------------------------------------------------
    # S1
    # Refrigerator exceeded maximum OFF duration
    # --------------------------------------------------------

    if rule == (
        "S1_CRITICAL_MAX_OFF_EXCEEDED"
    ):

        safe_action = "PRIORITY"


    # --------------------------------------------------------
    # S2
    # Refrigerator received DELAY
    # --------------------------------------------------------

    elif rule == (
        "S2_CRITICAL_APPLIANCE_DELAY"
    ):

        safe_action = "PRIORITY"


    # --------------------------------------------------------
    # S3
    # Running non-interruptible cycle
    # --------------------------------------------------------

    elif rule == (
        "S3_NON_INTERRUPTIBLE_CYCLE"
    ):

        safe_action = "TURN_ON"


    # --------------------------------------------------------
    # S4
    # Maximum allowed delay exceeded
    # --------------------------------------------------------

    elif rule == (
        "S4_MAX_DELAY_EXCEEDED"
    ):

        safe_action = "TURN_ON"


    else:

        safe_action = ml_action


    return pd.Series({

        "final_action":
            safe_action,

        "safety_override":
            int(
                safe_action
                != ml_action
            ),

        "safety_rule":
            rule,

        "original_violation":
            1
    })


# ============================================================
# 7. APPLY SAFETY LAYER TO NATURAL TEST DATA
# ============================================================

print(
    "\nApplying safety layer to unseen test data..."
)

natural_safety_results = (
    test_df.apply(
        lambda row:
        apply_safety_layer(
            row,
            row["ml_action"]
        ),
        axis=1
    )
)

natural_safety_results = (
    natural_safety_results
    .reset_index(drop=True)
)

test_df = (
    test_df
    .reset_index(drop=True)
)

test_df = pd.concat(
    [
        test_df,
        natural_safety_results
    ],
    axis=1
)


# ============================================================
# 8. CHECK FOR DUPLICATE COLUMNS
# ============================================================

natural_duplicates = (
    test_df.columns[
        test_df.columns.duplicated()
    ]
    .tolist()
)

if natural_duplicates:

    raise ValueError(
        "Duplicate columns detected in natural test data: "
        f"{natural_duplicates}"
    )

print(
    "PASS: Natural-test safety columns are unique."
)


# ============================================================
# 9. FINAL SAFETY VIOLATION CHECK
# ============================================================

def check_final_violation(
    row
):

    final_action = row[
        "final_action"
    ]

    violation, _ = (
        identify_safety_violation(
            row,
            final_action
        )
    )

    return int(
        violation
    )


test_df[
    "final_violation"
] = test_df.apply(
    check_final_violation,
    axis=1
)


# ============================================================
# 10. NATURAL TEST SAFETY METRICS
# ============================================================

total_test = len(
    test_df
)

unsafe_before = int(
    test_df[
        "original_violation"
    ].sum()
)

unsafe_after = int(
    test_df[
        "final_violation"
    ].sum()
)

override_count = int(
    test_df[
        "safety_override"
    ].sum()
)


unsafe_before_rate = (
    unsafe_before
    / total_test
    * 100
)

unsafe_after_rate = (
    unsafe_after
    / total_test
    * 100
)

override_rate = (
    override_count
    / total_test
    * 100
)

safety_compliance_before = (
    100
    - unsafe_before_rate
)

safety_compliance_after = (
    100
    - unsafe_after_rate
)


# ============================================================
# 11. CLASSIFICATION ACCURACY BEFORE / AFTER SAFETY
# ============================================================

ml_accuracy = accuracy_score(
    test_df[
        "target_action"
    ],
    test_df[
        "ml_action"
    ]
)

post_safety_accuracy = accuracy_score(
    test_df[
        "target_action"
    ],
    test_df[
        "final_action"
    ]
)


# ============================================================
# 12. SAVE NATURAL TEST SUMMARY
# ============================================================

natural_summary = pd.DataFrame([{

    "Total_Test_Records":
        total_test,

    "Unsafe_Recommendations_Before":
        unsafe_before,

    "Unsafe_Rate_Before_Percent":
        unsafe_before_rate,

    "Safety_Compliance_Before_Percent":
        safety_compliance_before,

    "Safety_Overrides":
        override_count,

    "Override_Rate_Percent":
        override_rate,

    "Unsafe_Actions_After":
        unsafe_after,

    "Unsafe_Rate_After_Percent":
        unsafe_after_rate,

    "Safety_Compliance_After_Percent":
        safety_compliance_after,

    "ML_Accuracy_Before_Safety":
        ml_accuracy,

    "Accuracy_After_Safety":
        post_safety_accuracy

}])


natural_summary.to_csv(
    os.path.join(
        RESULT_DIR,
        "natural_test_safety_summary.csv"
    ),
    index=False
)


# ============================================================
# 13. SAVE NATURAL SAFETY OVERRIDES
# ============================================================

override_columns = [

    "house_id",
    "Time",

    "appliance_category",
    "appliance_active",

    "off_duration_minutes",
    "max_off_minutes",
    "max_delay_minutes",

    "interruptible",

    "target_action",

    "ml_action",
    "final_action",

    "safety_rule"
]


existing_override_columns = [

    column
    for column in override_columns
    if column in test_df.columns

]


override_records = test_df[
    test_df[
        "safety_override"
    ] == 1
][
    existing_override_columns
].copy()


override_records.to_csv(
    os.path.join(
        RESULT_DIR,
        "natural_safety_overrides.csv"
    ),
    index=False
)


# ============================================================
# 14. SAFETY-RULE ACTIVATION SUMMARY
# ============================================================

unsafe_rows = test_df[
    test_df[
        "original_violation"
    ] == 1
]

if len(
    unsafe_rows
) > 0:

    rule_summary = (
        unsafe_rows[
            "safety_rule"
        ]
        .value_counts()
        .rename_axis(
            "Safety_Rule"
        )
        .reset_index(
            name="Activations"
        )
    )

else:

    rule_summary = pd.DataFrame(
        columns=[
            "Safety_Rule",
            "Activations"
        ]
    )


rule_summary.to_csv(
    os.path.join(
        RESULT_DIR,
        "safety_rule_activations.csv"
    ),
    index=False
)


# ============================================================
# 15. PREPARE CONTROLLED SAFETY STRESS TEST
# ============================================================

print(
    "\nCreating controlled safety stress-test scenarios..."
)

stress_parts = []


# ============================================================
# STRESS TYPE 1
# Critical refrigerator exceeds max OFF duration
# ============================================================

fridge_pool = test_df[
    test_df[
        "appliance_category"
    ] == "Refrigeration"
].copy()


if len(
    fridge_pool
) > 0:

    n = min(
        3000,
        len(fridge_pool)
    )

    fridge_stress = fridge_pool.sample(
        n=n,
        random_state=RANDOM_STATE
    ).copy()

    fridge_stress[
        "appliance_active"
    ] = 0

    fridge_stress[
        "previous_appliance_active"
    ] = 0

    multipliers = rng.uniform(
        1.0,
        1.5,
        size=len(
            fridge_stress
        )
    )

    fridge_stress[
        "off_duration_minutes"
    ] = (
        fridge_stress[
            "max_off_minutes"
        ]
        * multipliers
    )

    fridge_stress[
        "state_duration_minutes"
    ] = (
        fridge_stress[
            "off_duration_minutes"
        ]
    )

    fridge_stress[
        "on_duration_minutes"
    ] = 0

    fridge_stress[
        "stress_type"
    ] = (
        "Critical_Max_Off"
    )

    stress_parts.append(
        fridge_stress
    )


# ============================================================
# STRESS TYPE 2
# Running non-interruptible washing-machine cycle
# ============================================================

washer_pool = test_df[
    test_df[
        "appliance_category"
    ] == "Washing Machine"
].copy()


if len(
    washer_pool
) > 0:

    n = min(
        3000,
        len(washer_pool)
    )

    washer_stress = washer_pool.sample(
        n=n,
        random_state=RANDOM_STATE + 1
    ).copy()

    washer_stress[
        "appliance_active"
    ] = 1

    washer_stress[
        "previous_appliance_active"
    ] = 1

    washer_stress[
        "interruptible"
    ] = 0

    washer_stress[
        "on_duration_minutes"
    ] = rng.choice(
        [
            15,
            30,
            45,
            60
        ],
        size=len(
            washer_stress
        )
    )

    washer_stress[
        "off_duration_minutes"
    ] = 0

    washer_stress[
        "state_duration_minutes"
    ] = (
        washer_stress[
            "on_duration_minutes"
        ]
    )

    washer_stress[
        "stress_type"
    ] = (
        "Running_Washing_Machine"
    )

    stress_parts.append(
        washer_stress
    )


# ============================================================
# STRESS TYPE 3
# Running non-interruptible dishwasher cycle
# ============================================================

dish_pool = test_df[
    test_df[
        "appliance_category"
    ] == "Dishwasher"
].copy()


if len(
    dish_pool
) > 0:

    n = min(
        3000,
        len(dish_pool)
    )

    dish_stress = dish_pool.sample(
        n=n,
        random_state=RANDOM_STATE + 2
    ).copy()

    dish_stress[
        "appliance_active"
    ] = 1

    dish_stress[
        "previous_appliance_active"
    ] = 1

    dish_stress[
        "interruptible"
    ] = 0

    dish_stress[
        "on_duration_minutes"
    ] = rng.choice(
        [
            15,
            30,
            45,
            60
        ],
        size=len(
            dish_stress
        )
    )

    dish_stress[
        "off_duration_minutes"
    ] = 0

    dish_stress[
        "state_duration_minutes"
    ] = (
        dish_stress[
            "on_duration_minutes"
        ]
    )

    dish_stress[
        "stress_type"
    ] = (
        "Running_Dishwasher"
    )

    stress_parts.append(
        dish_stress
    )


# ============================================================
# STRESS TYPE 4
# Flexible appliance exceeds max allowed delay
# ============================================================

flex_pool = test_df[
    test_df[
        "appliance_category"
    ].isin(
        [
            "Washing Machine",
            "Dishwasher"
        ]
    )
].copy()


if len(
    flex_pool
) > 0:

    n = min(
        3000,
        len(flex_pool)
    )

    delay_stress = flex_pool.sample(
        n=n,
        random_state=RANDOM_STATE + 3
    ).copy()

    delay_stress[
        "appliance_active"
    ] = 0

    delay_stress[
        "previous_appliance_active"
    ] = 0

    multipliers = rng.uniform(
        1.0,
        1.3,
        size=len(
            delay_stress
        )
    )

    delay_stress[
        "off_duration_minutes"
    ] = (
        delay_stress[
            "max_delay_minutes"
        ]
        * multipliers
    )

    delay_stress[
        "state_duration_minutes"
    ] = (
        delay_stress[
            "off_duration_minutes"
        ]
    )

    delay_stress[
        "on_duration_minutes"
    ] = 0

    delay_stress[
        "stress_type"
    ] = (
        "Maximum_Delay_Exceeded"
    )

    stress_parts.append(
        delay_stress
    )


# ============================================================
# 16. COMBINE STRESS SCENARIOS
# ============================================================

if len(
    stress_parts
) == 0:

    raise ValueError(
        "No stress-test scenarios could be created."
    )


stress_df = pd.concat(
    stress_parts,
    ignore_index=True
)


print(
    f"Stress-test records: "
    f"{len(stress_df):,}"
)


# ============================================================
# 17. REMOVE PREVIOUS NATURAL-SAFETY RESULT COLUMNS
# ============================================================

# IMPORTANT:
#
# Stress rows were sampled from test_df.
# test_df already contains natural safety-evaluation outputs.
# Remove them before applying the stress-test safety layer.

previous_safety_columns = [

    "ml_action",
    "final_action",
    "safety_override",
    "safety_rule",
    "original_violation",
    "final_violation"

]


columns_to_remove = [

    column
    for column in previous_safety_columns
    if column in stress_df.columns

]


if columns_to_remove:

    stress_df = stress_df.drop(
        columns=columns_to_remove
    )


print(
    "\nRemoved inherited safety-result columns:"
)

print(
    columns_to_remove
)


# ============================================================
# 18. CHECK DUPLICATES BEFORE STRESS PREDICTION
# ============================================================

duplicates_before = (
    stress_df.columns[
        stress_df.columns.duplicated()
    ]
    .tolist()
)


if duplicates_before:

    raise ValueError(
        "Duplicate columns detected before stress prediction: "
        f"{duplicates_before}"
    )


print(
    "PASS: Stress-test columns are unique before prediction."
)


# ============================================================
# 19. GENERATE MODEL PREDICTIONS FOR STRESS SCENARIOS
# ============================================================

print(
    "\nGenerating ML recommendations for "
    "controlled stress-test scenarios..."
)

stress_predictions = model.predict(
    stress_df
)

stress_df[
    "ml_action"
] = label_encoder.inverse_transform(
    stress_predictions.astype(int)
)

print(
    "Stress-test ML recommendations generated."
)


# ============================================================
# 20. APPLY SAFETY LAYER TO STRESS DATA
# ============================================================

print(
    "Applying rule-based safety layer "
    "to stress-test scenarios..."
)

stress_safety_results = (
    stress_df.apply(
        lambda row:
        apply_safety_layer(
            row,
            row["ml_action"]
        ),
        axis=1
    )
)

stress_safety_results = (
    stress_safety_results
    .reset_index(drop=True)
)

stress_df = (
    stress_df
    .reset_index(drop=True)
)


stress_df = pd.concat(
    [
        stress_df,
        stress_safety_results
    ],
    axis=1
)


# ============================================================
# 21. CHECK DUPLICATES AFTER SAFETY LAYER
# ============================================================

duplicates_after = (
    stress_df.columns[
        stress_df.columns.duplicated()
    ]
    .tolist()
)


if duplicates_after:

    raise ValueError(
        "Duplicate columns detected after stress safety layer: "
        f"{duplicates_after}"
    )


print(
    "PASS: No duplicate safety columns detected."
)


# ============================================================
# 22. CHECK FINAL STRESS-TEST SAFETY
# ============================================================

stress_df[
    "final_violation"
] = stress_df.apply(
    check_final_violation,
    axis=1
)

print(
    "Stress-test final safety compliance checked."
)


# ============================================================
# 23. OVERALL STRESS-TEST METRICS
# ============================================================

stress_total = len(
    stress_df
)

stress_unsafe_before = int(
    stress_df[
        "original_violation"
    ].sum()
)

stress_unsafe_after = int(
    stress_df[
        "final_violation"
    ].sum()
)

stress_overrides = int(
    stress_df[
        "safety_override"
    ].sum()
)


stress_unsafe_before_percent = (
    stress_unsafe_before
    / stress_total
    * 100
)

stress_override_percent = (
    stress_overrides
    / stress_total
    * 100
)

stress_unsafe_after_percent = (
    stress_unsafe_after
    / stress_total
    * 100
)

stress_compliance_before = (
    100
    - stress_unsafe_before_percent
)

stress_compliance_after = (
    100
    - stress_unsafe_after_percent
)


# ============================================================
# 24. SAVE STRESS TEST SUMMARY
# ============================================================

stress_summary = pd.DataFrame([{

    "Stress_Records":
        stress_total,

    "Unsafe_Before":
        stress_unsafe_before,

    "Unsafe_Before_Percent":
        stress_unsafe_before_percent,

    "Safety_Compliance_Before_Percent":
        stress_compliance_before,

    "Safety_Overrides":
        stress_overrides,

    "Override_Percent":
        stress_override_percent,

    "Unsafe_After":
        stress_unsafe_after,

    "Unsafe_After_Percent":
        stress_unsafe_after_percent,

    "Final_Safety_Compliance_Percent":
        stress_compliance_after

}])


stress_summary.to_csv(
    os.path.join(
        RESULT_DIR,
        "stress_test_safety_summary.csv"
    ),
    index=False
)


# ============================================================
# 25. STRESS TEST BY TYPE
# ============================================================

stress_type_summary = (
    stress_df.groupby(
        "stress_type"
    )
    .agg(

        Records=(
            "stress_type",
            "size"
        ),

        Unsafe_Before=(
            "original_violation",
            "sum"
        ),

        Overrides=(
            "safety_override",
            "sum"
        ),

        Unsafe_After=(
            "final_violation",
            "sum"
        )

    )
    .reset_index()
)


stress_type_summary[
    "Unsafe_Before_Percent"
] = (
    stress_type_summary[
        "Unsafe_Before"
    ]
    /
    stress_type_summary[
        "Records"
    ]
    * 100
)


stress_type_summary[
    "Override_Percent"
] = (
    stress_type_summary[
        "Overrides"
    ]
    /
    stress_type_summary[
        "Records"
    ]
    * 100
)


stress_type_summary[
    "Safety_Compliance_After_Percent"
] = (
    1
    -
    (
        stress_type_summary[
            "Unsafe_After"
        ]
        /
        stress_type_summary[
            "Records"
        ]
    )
) * 100


stress_type_summary.to_csv(
    os.path.join(
        RESULT_DIR,
        "stress_test_by_type.csv"
    ),
    index=False
)


# ============================================================
# 26. SAVE STRESS TEST RECORDS
# ============================================================

stress_save_columns = [

    "stress_type",
    "house_id",

    "appliance_category",
    "appliance_active",

    "off_duration_minutes",
    "max_off_minutes",
    "max_delay_minutes",

    "interruptible",

    "ml_action",
    "final_action",

    "safety_rule",

    "original_violation",
    "safety_override",
    "final_violation"
]


existing_stress_columns = [

    column
    for column in stress_save_columns
    if column in stress_df.columns

]


stress_df[
    existing_stress_columns
].to_csv(
    os.path.join(
        RESULT_DIR,
        "stress_test_records.csv"
    ),
    index=False
)


# ============================================================
# 27. NATURAL SAFETY COMPLIANCE PLOT
# ============================================================

labels = [
    "Before Safety Layer",
    "After Safety Layer"
]

natural_values = [
    safety_compliance_before,
    safety_compliance_after
]


fig, ax = plt.subplots(
    figsize=(8, 6)
)

ax.bar(
    labels,
    natural_values
)

ax.set_ylim(
    0,
    105
)

ax.set_ylabel(
    "Safety Compliance (%)"
)

ax.set_title(
    "Safety Compliance on Unseen Test Households"
)


for index, value in enumerate(
    natural_values
):

    ax.text(
        index,
        min(
            value + 1,
            103
        ),
        f"{value:.2f}%",
        ha="center"
    )


fig.tight_layout()


plt.savefig(
    os.path.join(
        PLOT_DIR,
        "natural_safety_compliance.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 28. STRESS-TEST SAFETY COMPLIANCE PLOT
# ============================================================

stress_values = [
    stress_compliance_before,
    stress_compliance_after
]


fig, ax = plt.subplots(
    figsize=(8, 6)
)

ax.bar(
    [
        "Before Safety Layer",
        "After Safety Layer"
    ],
    stress_values
)

ax.set_ylim(
    0,
    105
)

ax.set_ylabel(
    "Safety Compliance (%)"
)

ax.set_title(
    "Controlled Safety Stress-Test Compliance"
)


for index, value in enumerate(
    stress_values
):

    ax.text(
        index,
        min(
            value + 1,
            103
        ),
        f"{value:.2f}%",
        ha="center"
    )


fig.tight_layout()


plt.savefig(
    os.path.join(
        PLOT_DIR,
        "stress_test_safety_compliance.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 29. SAFETY RULE ACTIVATION PLOT
# ============================================================

if len(
    rule_summary
) > 0:

    plot_data = (
        rule_summary
        .sort_values(
            "Activations",
            ascending=True
        )
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.barh(
        plot_data[
            "Safety_Rule"
        ],
        plot_data[
            "Activations"
        ]
    )

    ax.set_xlabel(
        "Number of Activations"
    )

    ax.set_title(
        "Safety Rule Activations on Unseen Test Data"
    )

    fig.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "safety_rule_activations.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 30. FINAL TERMINAL REPORT
# ============================================================

print("\n")
print("============================================")
print("NATURAL TEST SAFETY RESULTS")
print("============================================")

print(
    f"Total test records: "
    f"{total_test:,}"
)

print(
    f"Unsafe ML recommendations: "
    f"{unsafe_before:,}"
)

print(
    f"Unsafe rate before safety layer: "
    f"{unsafe_before_rate:.2f}%"
)

print(
    f"Safety compliance before: "
    f"{safety_compliance_before:.2f}%"
)

print(
    f"Safety overrides: "
    f"{override_count:,}"
)

print(
    f"Override rate: "
    f"{override_rate:.2f}%"
)

print(
    f"Unsafe actions after safety layer: "
    f"{unsafe_after:,}"
)

print(
    f"Unsafe rate after safety layer: "
    f"{unsafe_after_rate:.2f}%"
)

print(
    f"Safety compliance after: "
    f"{safety_compliance_after:.2f}%"
)

print(
    f"ML classification accuracy before safety: "
    f"{ml_accuracy:.4f}"
)

print(
    f"Classification accuracy after safety: "
    f"{post_safety_accuracy:.4f}"
)


print("\nSafety-rule activations:")

if len(
    rule_summary
) > 0:

    print(
        rule_summary.to_string(
            index=False
        )
    )

else:

    print(
        "No natural safety-rule activations."
    )


print("\n")
print("============================================")
print("CONTROLLED STRESS TEST RESULTS")
print("============================================")

print(
    f"Stress scenarios: "
    f"{stress_total:,}"
)

print(
    f"Unsafe before safety layer: "
    f"{stress_unsafe_before:,}"
)

print(
    f"Unsafe rate before: "
    f"{stress_unsafe_before_percent:.2f}%"
)

print(
    f"Safety compliance before: "
    f"{stress_compliance_before:.2f}%"
)

print(
    f"Safety overrides: "
    f"{stress_overrides:,}"
)

print(
    f"Override rate: "
    f"{stress_override_percent:.2f}%"
)

print(
    f"Unsafe after safety layer: "
    f"{stress_unsafe_after:,}"
)

print(
    f"Unsafe rate after: "
    f"{stress_unsafe_after_percent:.2f}%"
)

print(
    f"Final stress-test safety compliance: "
    f"{stress_compliance_after:.2f}%"
)


print(
    "\nStress-test results by scenario:"
)

print(
    stress_type_summary.to_string(
        index=False
    )
)


print("\n============================================")
print("GENERATED OUTPUT")
print("============================================")

print(
    f"Results: {RESULT_DIR}"
)

print(
    f"Plots: {PLOT_DIR}"
)

print(
    "\nSafety-layer evaluation completed."
)