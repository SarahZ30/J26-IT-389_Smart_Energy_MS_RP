import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit


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

OUTPUT_DIR = "operational_benefit_output"

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

INTERVAL_HOURS = 0.25  # 15-minute REFIT interval


# ============================================================
# OPERATIONAL SIMULATION ASSUMPTIONS
# ============================================================

# IMPORTANT:
#
# These are PP1 experimental simulation assumptions.
# They are NOT guaranteed real-world appliance savings.
#
# Final deployment should replace them with:
# - appliance-specific measured power profiles
# - official tariff values
# - manufacturer operating constraints
# - live IoT measurements


# ECO_MODE:
# Assume approximately 15% reduction in current-interval energy.
ECO_REDUCTION_FACTOR = 0.15


# LOAD_BALANCE:
# Assume 25% of current interval energy is shifted away
# from the current high-demand interval.
LOAD_BALANCE_SHIFT_FACTOR = 0.25


# DELAY:
# Assume energy that would otherwise occur now is moved
# to an off-peak interval.
DELAY_SHIFT_FACTOR = 1.00


# Experimental normalized tariff-cost index
OFF_PEAK_COST_INDEX = 0.30


# ============================================================
# 1. START
# ============================================================

print("============================================")
print("COMPONENT 3 - OPERATIONAL BENEFIT EVALUATION")
print("============================================")


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading Component 3 V2 dataset...")

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
# 3. LOAD SELECTED MODEL
# ============================================================

print(
    "\nLoading selected XGBoost model..."
)

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
    "Selected configuration: "
    "XGBoost + Personalization"
)


# ============================================================
# 4. RECREATE UNSEEN HOUSEHOLD TEST SET
# ============================================================

groups = df[
    "house_id"
]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_val_idx, test_idx = next(
    splitter.split(
        df,
        df["target_action"],
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


print("\nTest dataset:")

print(
    f"Records: {len(test_df):,}"
)

print(
    "Unseen houses:"
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
# 5. MODEL PREDICTIONS
# ============================================================

print(
    "\nGenerating scheduling recommendations..."
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
    "Scheduling recommendations generated."
)


# ============================================================
# 6. SAFETY RULE IDENTIFICATION
# ============================================================

def identify_safety_violation(
    row,
    action
):

    action = str(
        action
    )

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


    # --------------------------------------------------------
    # S1 - Critical refrigeration maximum OFF duration
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # S2 - Critical refrigeration must not be delayed
    # --------------------------------------------------------

    if (
        appliance == "Refrigeration"
        and action == "DELAY"
    ):

        return (
            True,
            "S2_CRITICAL_APPLIANCE_DELAY"
        )


    # --------------------------------------------------------
    # S3 - Non-interruptible active cycle
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # S4 - Maximum delay exceeded
    # --------------------------------------------------------

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
# 7. SAFETY-LAYER OVERRIDE
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


    if not violation:

        return pd.Series({

            "final_action":
                ml_action,

            "safety_override":
                0,

            "safety_rule":
                "NONE"

        })


    if rule in [
        "S1_CRITICAL_MAX_OFF_EXCEEDED",
        "S2_CRITICAL_APPLIANCE_DELAY"
    ]:

        safe_action = "PRIORITY"


    elif rule in [
        "S3_NON_INTERRUPTIBLE_CYCLE",
        "S4_MAX_DELAY_EXCEEDED"
    ]:

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
            rule

    })


# ============================================================
# 8. APPLY SAFETY LAYER
# ============================================================

print(
    "\nApplying rule-based safety layer..."
)

safety_results = test_df.apply(
    lambda row:
    apply_safety_layer(
        row,
        row["ml_action"]
    ),
    axis=1
)

test_df = pd.concat(
    [
        test_df,
        safety_results.reset_index(
            drop=True
        )
    ],
    axis=1
)

print(
    f"Safety overrides: "
    f"{test_df['safety_override'].sum():,}"
)


# ============================================================
# 9. BASELINE ENERGY
# ============================================================

print(
    "\nCalculating baseline energy..."
)


# appliance_mean_w is average appliance power
# over approximately 15 minutes.
#
# Energy:
#
# kWh = W × hours / 1000

test_df[
    "baseline_energy_kwh"
] = (
    test_df[
        "appliance_mean_w"
    ]
    .clip(lower=0)
    * INTERVAL_HOURS
    / 1000
)


# ============================================================
# 10. BASELINE NORMALIZED COST
# ============================================================

test_df[
    "baseline_cost_index"
] = (
    test_df[
        "baseline_energy_kwh"
    ]
    * test_df[
        "tariff_cost_index"
    ]
)


# ============================================================
# 11. ACTION EFFECT SIMULATION
# ============================================================

def simulate_action_effect(
    row,
    action
):

    action = str(
        action
    )

    baseline_energy = float(
        row[
            "baseline_energy_kwh"
        ]
    )

    current_tariff = float(
        row[
            "tariff_cost_index"
        ]
    )

    active = int(
        row[
            "appliance_active"
        ]
    )


    # --------------------------------------------------------
    # Default:
    # no change
    # --------------------------------------------------------

    current_energy = (
        baseline_energy
    )

    shifted_energy = 0.0

    avoided_energy = 0.0

    eco_saved_energy = 0.0


    # ========================================================
    # TURN OFF
    # ========================================================

    if action == "TURN_OFF":

        if active == 1:

            avoided_energy = (
                baseline_energy
            )

            current_energy = 0.0

        else:

            current_energy = (
                baseline_energy
            )


    # ========================================================
    # DELAY
    # ========================================================

    elif action == "DELAY":

        if baseline_energy > 0:

            shifted_energy = (
                baseline_energy
                * DELAY_SHIFT_FACTOR
            )

            current_energy = (
                baseline_energy
                - shifted_energy
            )


    # ========================================================
    # ECO MODE
    # ========================================================

    elif action == "ECO_MODE":

        if baseline_energy > 0:

            eco_saved_energy = (
                baseline_energy
                * ECO_REDUCTION_FACTOR
            )

            current_energy = (
                baseline_energy
                - eco_saved_energy
            )


    # ========================================================
    # LOAD BALANCE
    # ========================================================

    elif action == "LOAD_BALANCE":

        if baseline_energy > 0:

            shifted_energy = (
                baseline_energy
                * LOAD_BALANCE_SHIFT_FACTOR
            )

            current_energy = (
                baseline_energy
                - shifted_energy
            )


    # ========================================================
    # TURN ON / PRIORITY
    # ========================================================

    elif action in [
        "TURN_ON",
        "PRIORITY"
    ]:

        # Conservative assumption:
        #
        # Do not invent additional energy for records where
        # the historical appliance was inactive.
        #
        # Keep observed baseline energy unchanged.

        current_energy = (
            baseline_energy
        )


    # ========================================================
    # COST AFTER SCHEDULING
    # ========================================================

    current_cost = (
        current_energy
        * current_tariff
    )

    shifted_cost = (
        shifted_energy
        * OFF_PEAK_COST_INDEX
    )

    scheduled_cost = (
        current_cost
        + shifted_cost
    )


    return pd.Series({

        "scheduled_current_energy_kwh":
            current_energy,

        "shifted_energy_kwh":
            shifted_energy,

        "avoided_energy_kwh":
            avoided_energy,

        "eco_saved_energy_kwh":
            eco_saved_energy,

        "scheduled_cost_index":
            scheduled_cost

    })


# ============================================================
# 12. RAW ML OPERATIONAL EFFECT
# ============================================================

print(
    "Simulating raw ML operational effect..."
)

raw_effect = test_df.apply(
    lambda row:
    simulate_action_effect(
        row,
        row["ml_action"]
    ),
    axis=1
)

raw_effect = raw_effect.add_prefix(
    "raw_ml_"
)

test_df = pd.concat(
    [
        test_df,
        raw_effect
    ],
    axis=1
)


# ============================================================
# 13. SAFETY-ADJUSTED OPERATIONAL EFFECT
# ============================================================

print(
    "Simulating safety-adjusted operational effect..."
)

final_effect = test_df.apply(
    lambda row:
    simulate_action_effect(
        row,
        row["final_action"]
    ),
    axis=1
)

final_effect = final_effect.add_prefix(
    "final_"
)

test_df = pd.concat(
    [
        test_df,
        final_effect
    ],
    axis=1
)


# ============================================================
# 14. CURRENT-INTERVAL POWER ESTIMATE
# ============================================================

# Convert scheduled current energy back to mean W:
#
# W = kWh × 1000 / hours

test_df[
    "baseline_interval_power_w"
] = (
    test_df[
        "baseline_energy_kwh"
    ]
    * 1000
    / INTERVAL_HOURS
)

test_df[
    "final_interval_power_w"
] = (
    test_df[
        "final_scheduled_current_energy_kwh"
    ]
    * 1000
    / INTERVAL_HOURS
)


# ============================================================
# 15. COMFORT SCORE
# ============================================================

def calculate_comfort_score(
    row,
    action
):

    action = str(
        action
    )

    flexibility = float(
        row[
            "user_flexibility"
        ]
    )

    occupancy = int(
        row[
            "occupancy"
        ]
    )

    active = int(
        row[
            "appliance_active"
        ]
    )


    if action == "PRIORITY":

        score = 98.0


    elif action == "TURN_ON":

        score = 95.0


    elif action == "ECO_MODE":

        score = (
            85.0
            + 5.0 * flexibility
        )


    elif action == "LOAD_BALANCE":

        score = (
            78.0
            + 10.0 * flexibility
        )


    elif action == "DELAY":

        score = (
            65.0
            + 25.0 * flexibility
        )

        if occupancy == 1:

            score -= 5.0


    elif action == "TURN_OFF":

        if active == 0:

            score = 92.0

        else:

            score = (
                55.0
                + 20.0 * flexibility
            )

            if occupancy == 1:

                score -= 10.0


    else:

        score = 75.0


    return float(
        np.clip(
            score,
            0,
            100
        )
    )


test_df[
    "comfort_score"
] = test_df.apply(
    lambda row:
    calculate_comfort_score(
        row,
        row[
            "final_action"
        ]
    ),
    axis=1
)


# ============================================================
# 16. BENEFIT COLUMNS
# ============================================================

test_df[
    "cost_index_reduction"
] = (
    test_df[
        "baseline_cost_index"
    ]
    -
    test_df[
        "final_scheduled_cost_index"
    ]
)


test_df[
    "cost_index_reduction_percent"
] = np.where(

    test_df[
        "baseline_cost_index"
    ] > 0,

    (
        test_df[
            "cost_index_reduction"
        ]
        /
        test_df[
            "baseline_cost_index"
        ]
        * 100
    ),

    0

)


test_df[
    "current_interval_energy_reduction_kwh"
] = (

    test_df[
        "baseline_energy_kwh"
    ]

    -

    test_df[
        "final_scheduled_current_energy_kwh"
    ]

)


test_df[
    "current_interval_power_reduction_w"
] = (

    test_df[
        "baseline_interval_power_w"
    ]

    -

    test_df[
        "final_interval_power_w"
    ]

)


# ============================================================
# 17. PEAK PERIOD SUBSET
# ============================================================

peak_df = test_df[
    test_df[
        "tariff_band"
    ] == "Peak"
].copy()


# ============================================================
# 18. OVERALL METRICS
# ============================================================

baseline_energy_total = (
    test_df[
        "baseline_energy_kwh"
    ].sum()
)

scheduled_current_energy_total = (
    test_df[
        "final_scheduled_current_energy_kwh"
    ].sum()
)

shifted_energy_total = (
    test_df[
        "final_shifted_energy_kwh"
    ].sum()
)

avoided_energy_total = (
    test_df[
        "final_avoided_energy_kwh"
    ].sum()
)

eco_saved_energy_total = (
    test_df[
        "final_eco_saved_energy_kwh"
    ].sum()
)


baseline_cost_total = (
    test_df[
        "baseline_cost_index"
    ].sum()
)

scheduled_cost_total = (
    test_df[
        "final_scheduled_cost_index"
    ].sum()
)


cost_reduction_total = (
    baseline_cost_total
    - scheduled_cost_total
)


if baseline_cost_total > 0:

    cost_reduction_percent = (
        cost_reduction_total
        / baseline_cost_total
        * 100
    )

else:

    cost_reduction_percent = 0.0


average_comfort = (
    test_df[
        "comfort_score"
    ].mean()
)


# ============================================================
# 19. PEAK METRICS
# ============================================================

peak_baseline_energy = (
    peak_df[
        "baseline_energy_kwh"
    ].sum()
)

peak_current_scheduled_energy = (
    peak_df[
        "final_scheduled_current_energy_kwh"
    ].sum()
)


peak_energy_reduction = (
    peak_baseline_energy
    - peak_current_scheduled_energy
)


if peak_baseline_energy > 0:

    peak_energy_reduction_percent = (
        peak_energy_reduction
        / peak_baseline_energy
        * 100
    )

else:

    peak_energy_reduction_percent = (
        0.0
    )


average_peak_baseline_power = (
    peak_df[
        "baseline_interval_power_w"
    ].mean()
)

average_peak_scheduled_power = (
    peak_df[
        "final_interval_power_w"
    ].mean()
)


if average_peak_baseline_power > 0:

    average_peak_power_reduction_percent = (

        (
            average_peak_baseline_power
            -
            average_peak_scheduled_power
        )

        / average_peak_baseline_power

        * 100
    )

else:

    average_peak_power_reduction_percent = (
        0.0
    )


# ============================================================
# 20. ACTION COUNTS
# ============================================================

action_counts = (
    test_df[
        "final_action"
    ]
    .value_counts()
)


delay_count = int(
    action_counts.get(
        "DELAY",
        0
    )
)

eco_count = int(
    action_counts.get(
        "ECO_MODE",
        0
    )
)

load_balance_count = int(
    action_counts.get(
        "LOAD_BALANCE",
        0
    )
)

turn_off_count = int(
    action_counts.get(
        "TURN_OFF",
        0
    )
)

priority_count = int(
    action_counts.get(
        "PRIORITY",
        0
    )
)

turn_on_count = int(
    action_counts.get(
        "TURN_ON",
        0
    )
)


# ============================================================
# 21. OVERALL SUMMARY
# ============================================================

overall_summary = pd.DataFrame([{

    "Evaluation_Type":
        "PP1_Simulation_Unseen_Households",

    "Test_Records":
        len(test_df),

    "Baseline_Energy_kWh":
        baseline_energy_total,

    "Scheduled_Current_Interval_Energy_kWh":
        scheduled_current_energy_total,

    "Energy_Shifted_kWh":
        shifted_energy_total,

    "Energy_Avoided_kWh":
        avoided_energy_total,

    "Eco_Mode_Energy_Reduction_kWh":
        eco_saved_energy_total,

    "Baseline_Normalized_Cost_Index":
        baseline_cost_total,

    "Scheduled_Normalized_Cost_Index":
        scheduled_cost_total,

    "Normalized_Cost_Index_Reduction":
        cost_reduction_total,

    "Normalized_Cost_Index_Reduction_Percent":
        cost_reduction_percent,

    "Peak_Baseline_Energy_kWh":
        peak_baseline_energy,

    "Peak_Scheduled_Current_Energy_kWh":
        peak_current_scheduled_energy,

    "Peak_Period_Energy_Reduction_kWh":
        peak_energy_reduction,

    "Peak_Period_Energy_Reduction_Percent":
        peak_energy_reduction_percent,

    "Average_Peak_Baseline_Power_W":
        average_peak_baseline_power,

    "Average_Peak_Scheduled_Power_W":
        average_peak_scheduled_power,

    "Average_Peak_Power_Reduction_Percent":
        average_peak_power_reduction_percent,

    "Average_User_Comfort_Score":
        average_comfort,

    "Safety_Overrides":
        int(
            test_df[
                "safety_override"
            ].sum()
        ),

    "Delay_Actions":
        delay_count,

    "Eco_Mode_Actions":
        eco_count,

    "Load_Balance_Actions":
        load_balance_count,

    "Turn_Off_Actions":
        turn_off_count,

    "Turn_On_Actions":
        turn_on_count,

    "Priority_Actions":
        priority_count

}])


overall_summary.to_csv(
    os.path.join(
        RESULT_DIR,
        "overall_benefit_summary.csv"
    ),
    index=False
)


# ============================================================
# 22. BENEFIT BY ACTION
# ============================================================

benefit_by_action = (
    test_df.groupby(
        "final_action"
    )
    .agg(

        Records=(
            "final_action",
            "size"
        ),

        Baseline_Energy_kWh=(
            "baseline_energy_kwh",
            "sum"
        ),

        Scheduled_Current_Energy_kWh=(
            "final_scheduled_current_energy_kwh",
            "sum"
        ),

        Shifted_Energy_kWh=(
            "final_shifted_energy_kwh",
            "sum"
        ),

        Avoided_Energy_kWh=(
            "final_avoided_energy_kwh",
            "sum"
        ),

        Eco_Saved_Energy_kWh=(
            "final_eco_saved_energy_kwh",
            "sum"
        ),

        Baseline_Cost_Index=(
            "baseline_cost_index",
            "sum"
        ),

        Scheduled_Cost_Index=(
            "final_scheduled_cost_index",
            "sum"
        ),

        Average_Comfort_Score=(
            "comfort_score",
            "mean"
        )

    )
    .reset_index()
)


benefit_by_action[
    "Cost_Index_Reduction"
] = (
    benefit_by_action[
        "Baseline_Cost_Index"
    ]
    -
    benefit_by_action[
        "Scheduled_Cost_Index"
    ]
)


benefit_by_action.to_csv(
    os.path.join(
        RESULT_DIR,
        "benefit_by_action.csv"
    ),
    index=False
)


# ============================================================
# 23. BENEFIT BY APPLIANCE
# ============================================================

benefit_by_appliance = (
    test_df.groupby(
        "appliance_category"
    )
    .agg(

        Records=(
            "appliance_category",
            "size"
        ),

        Baseline_Energy_kWh=(
            "baseline_energy_kwh",
            "sum"
        ),

        Scheduled_Current_Energy_kWh=(
            "final_scheduled_current_energy_kwh",
            "sum"
        ),

        Shifted_Energy_kWh=(
            "final_shifted_energy_kwh",
            "sum"
        ),

        Avoided_Energy_kWh=(
            "final_avoided_energy_kwh",
            "sum"
        ),

        Eco_Saved_Energy_kWh=(
            "final_eco_saved_energy_kwh",
            "sum"
        ),

        Baseline_Cost_Index=(
            "baseline_cost_index",
            "sum"
        ),

        Scheduled_Cost_Index=(
            "final_scheduled_cost_index",
            "sum"
        ),

        Average_Comfort_Score=(
            "comfort_score",
            "mean"
        )

    )
    .reset_index()
)


benefit_by_appliance[
    "Cost_Index_Reduction"
] = (
    benefit_by_appliance[
        "Baseline_Cost_Index"
    ]
    -
    benefit_by_appliance[
        "Scheduled_Cost_Index"
    ]
)


benefit_by_appliance[
    "Cost_Index_Reduction_Percent"
] = np.where(

    benefit_by_appliance[
        "Baseline_Cost_Index"
    ] > 0,

    (
        benefit_by_appliance[
            "Cost_Index_Reduction"
        ]
        /
        benefit_by_appliance[
            "Baseline_Cost_Index"
        ]
        * 100
    ),

    0

)


benefit_by_appliance.to_csv(
    os.path.join(
        RESULT_DIR,
        "benefit_by_appliance.csv"
    ),
    index=False
)


# ============================================================
# 24. BENEFIT BY USER PROFILE
# ============================================================

benefit_by_profile = (
    test_df.groupby(
        "user_profile"
    )
    .agg(

        Records=(
            "user_profile",
            "size"
        ),

        Baseline_Energy_kWh=(
            "baseline_energy_kwh",
            "sum"
        ),

        Scheduled_Current_Energy_kWh=(
            "final_scheduled_current_energy_kwh",
            "sum"
        ),

        Shifted_Energy_kWh=(
            "final_shifted_energy_kwh",
            "sum"
        ),

        Avoided_Energy_kWh=(
            "final_avoided_energy_kwh",
            "sum"
        ),

        Baseline_Cost_Index=(
            "baseline_cost_index",
            "sum"
        ),

        Scheduled_Cost_Index=(
            "final_scheduled_cost_index",
            "sum"
        ),

        Average_Comfort_Score=(
            "comfort_score",
            "mean"
        )

    )
    .reset_index()
)


benefit_by_profile[
    "Cost_Index_Reduction"
] = (
    benefit_by_profile[
        "Baseline_Cost_Index"
    ]
    -
    benefit_by_profile[
        "Scheduled_Cost_Index"
    ]
)


benefit_by_profile[
    "Cost_Index_Reduction_Percent"
] = np.where(

    benefit_by_profile[
        "Baseline_Cost_Index"
    ] > 0,

    (
        benefit_by_profile[
            "Cost_Index_Reduction"
        ]
        /
        benefit_by_profile[
            "Baseline_Cost_Index"
        ]
        * 100
    ),

    0

)


benefit_by_profile.to_csv(
    os.path.join(
        RESULT_DIR,
        "benefit_by_user_profile.csv"
    ),
    index=False
)


# ============================================================
# 25. PEAK-PERIOD ANALYSIS
# ============================================================

peak_period_analysis = (
    peak_df.groupby(
        "final_action"
    )
    .agg(

        Records=(
            "final_action",
            "size"
        ),

        Baseline_Energy_kWh=(
            "baseline_energy_kwh",
            "sum"
        ),

        Scheduled_Current_Energy_kWh=(
            "final_scheduled_current_energy_kwh",
            "sum"
        ),

        Shifted_Energy_kWh=(
            "final_shifted_energy_kwh",
            "sum"
        ),

        Baseline_Cost_Index=(
            "baseline_cost_index",
            "sum"
        ),

        Scheduled_Cost_Index=(
            "final_scheduled_cost_index",
            "sum"
        )

    )
    .reset_index()
)


peak_period_analysis[
    "Current_Interval_Energy_Reduction_kWh"
] = (

    peak_period_analysis[
        "Baseline_Energy_kWh"
    ]

    -

    peak_period_analysis[
        "Scheduled_Current_Energy_kWh"
    ]

)


peak_period_analysis.to_csv(
    os.path.join(
        RESULT_DIR,
        "peak_period_analysis.csv"
    ),
    index=False
)


# ============================================================
# 26. SAFETY ADJUSTMENT IMPACT
# ============================================================

raw_ml_cost_total = (
    test_df[
        "raw_ml_scheduled_cost_index"
    ].sum()
)

final_cost_total = (
    test_df[
        "final_scheduled_cost_index"
    ].sum()
)


raw_ml_shifted_energy = (
    test_df[
        "raw_ml_shifted_energy_kwh"
    ].sum()
)

final_shifted_energy = (
    test_df[
        "final_shifted_energy_kwh"
    ].sum()
)


safety_adjustment_summary = pd.DataFrame([{

    "Raw_ML_Scheduled_Cost_Index":
        raw_ml_cost_total,

    "Safety_Adjusted_Cost_Index":
        final_cost_total,

    "Cost_Index_Change_Due_To_Safety":
        (
            final_cost_total
            - raw_ml_cost_total
        ),

    "Raw_ML_Shifted_Energy_kWh":
        raw_ml_shifted_energy,

    "Safety_Adjusted_Shifted_Energy_kWh":
        final_shifted_energy,

    "Shifted_Energy_Change_Due_To_Safety_kWh":
        (
            final_shifted_energy
            - raw_ml_shifted_energy
        ),

    "Safety_Overrides":
        int(
            test_df[
                "safety_override"
            ].sum()
        )

}])


safety_adjustment_summary.to_csv(
    os.path.join(
        RESULT_DIR,
        "safety_adjustment_impact.csv"
    ),
    index=False
)


# ============================================================
# 27. SAVE RECORD-LEVEL RESULTS SAMPLE
# ============================================================

sample_columns = [

    "house_id",
    "Time",

    "appliance_category",

    "user_profile",
    "user_flexibility",

    "tariff_band",
    "tariff_cost_index",

    "predicted_load_level",

    "appliance_active",
    "appliance_mean_w",

    "ml_action",
    "final_action",

    "safety_override",
    "safety_rule",

    "baseline_energy_kwh",

    "final_scheduled_current_energy_kwh",

    "final_shifted_energy_kwh",

    "final_avoided_energy_kwh",

    "final_eco_saved_energy_kwh",

    "baseline_cost_index",

    "final_scheduled_cost_index",

    "cost_index_reduction",

    "comfort_score"

]


existing_sample_columns = [

    column
    for column in sample_columns
    if column in test_df.columns

]


sample_n = min(
    5000,
    len(test_df)
)


test_df[
    existing_sample_columns
].sample(
    n=sample_n,
    random_state=RANDOM_STATE
).to_csv(
    os.path.join(
        RESULT_DIR,
        "operational_benefit_sample.csv"
    ),
    index=False
)


# ============================================================
# 28. PLOT - BASELINE VS SCHEDULED CURRENT ENERGY
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 6)
)

ax.bar(
    [
        "Baseline",
        "Scheduled\nCurrent Interval"
    ],
    [
        baseline_energy_total,
        scheduled_current_energy_total
    ]
)

ax.set_ylabel(
    "Energy (kWh)"
)

ax.set_title(
    "Baseline vs Scheduled Current-Interval Energy"
)

fig.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "baseline_vs_scheduled_energy.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 29. PLOT - PEAK PERIOD ENERGY
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 6)
)

ax.bar(
    [
        "Baseline Peak",
        "Scheduled Peak"
    ],
    [
        peak_baseline_energy,
        peak_current_scheduled_energy
    ]
)

ax.set_ylabel(
    "Peak-Period Energy (kWh)"
)

ax.set_title(
    "Peak-Period Energy Exposure"
)

fig.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "peak_load_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 30. PLOT - NORMALIZED COST
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 6)
)

ax.bar(
    [
        "Baseline",
        "Scheduled"
    ],
    [
        baseline_cost_total,
        scheduled_cost_total
    ]
)

ax.set_ylabel(
    "Normalized Cost Index"
)

ax.set_title(
    "Estimated Tariff-Cost Index Comparison"
)

fig.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "estimated_cost_index_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 31. PLOT - APPLIANCE COST BENEFIT
# ============================================================

plot_appliance = (
    benefit_by_appliance
    .sort_values(
        "Cost_Index_Reduction_Percent",
        ascending=True
    )
)

fig, ax = plt.subplots(
    figsize=(9, 6)
)

ax.barh(
    plot_appliance[
        "appliance_category"
    ],
    plot_appliance[
        "Cost_Index_Reduction_Percent"
    ]
)

ax.set_xlabel(
    "Estimated Cost-Index Reduction (%)"
)

ax.set_title(
    "Estimated Operational Benefit by Appliance"
)

fig.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "appliance_benefits.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 32. PLOT - USER PROFILE BENEFIT
# ============================================================

plot_profile = (
    benefit_by_profile
    .sort_values(
        "Cost_Index_Reduction_Percent",
        ascending=True
    )
)

fig, ax = plt.subplots(
    figsize=(9, 6)
)

ax.barh(
    plot_profile[
        "user_profile"
    ],
    plot_profile[
        "Cost_Index_Reduction_Percent"
    ]
)

ax.set_xlabel(
    "Estimated Cost-Index Reduction (%)"
)

ax.set_title(
    "Estimated Benefit by User Profile"
)

fig.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "user_profile_benefits.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 33. PLOT - COMFORT BY USER PROFILE
# ============================================================

comfort_by_profile = (
    test_df.groupby(
        "user_profile"
    )[
        "comfort_score"
    ]
    .mean()
    .sort_values()
)


fig, ax = plt.subplots(
    figsize=(9, 6)
)

ax.barh(
    comfort_by_profile.index,
    comfort_by_profile.values
)

ax.set_xlim(
    0,
    100
)

ax.set_xlabel(
    "Average Comfort Score"
)

ax.set_title(
    "Estimated User Comfort by Profile"
)

fig.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "user_profile_comfort.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 34. PLOT - ACTION DISTRIBUTION
# ============================================================

action_distribution = (
    test_df[
        "final_action"
    ]
    .value_counts()
)


fig, ax = plt.subplots(
    figsize=(9, 6)
)

ax.bar(
    action_distribution.index,
    action_distribution.values
)

ax.set_ylabel(
    "Number of Decisions"
)

ax.set_title(
    "Safety-Adjusted Scheduling Action Distribution"
)

ax.tick_params(
    axis="x",
    rotation=30
)

fig.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "final_action_distribution.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 35. FINAL TERMINAL REPORT
# ============================================================

print("\n")
print("============================================")
print("OPERATIONAL BENEFIT RESULTS")
print("============================================")


print(
    f"\nTest records: "
    f"{len(test_df):,}"
)


print("\nENERGY")

print(
    f"Baseline observed energy: "
    f"{baseline_energy_total:.4f} kWh"
)

print(
    f"Scheduled current-interval energy: "
    f"{scheduled_current_energy_total:.4f} kWh"
)

print(
    f"Energy shifted to later/off-peak periods: "
    f"{shifted_energy_total:.4f} kWh"
)

print(
    f"Estimated avoided interval energy: "
    f"{avoided_energy_total:.4f} kWh"
)

print(
    f"Estimated ECO-mode reduction: "
    f"{eco_saved_energy_total:.4f} kWh"
)


print("\nNORMALIZED COST")

print(
    f"Baseline normalized cost index: "
    f"{baseline_cost_total:.4f}"
)

print(
    f"Scheduled normalized cost index: "
    f"{scheduled_cost_total:.4f}"
)

print(
    f"Estimated cost-index reduction: "
    f"{cost_reduction_total:.4f}"
)

print(
    f"Estimated cost-index reduction: "
    f"{cost_reduction_percent:.2f}%"
)


print("\nPEAK PERIOD")

print(
    f"Baseline peak-period energy: "
    f"{peak_baseline_energy:.4f} kWh"
)

print(
    f"Scheduled peak-period current energy: "
    f"{peak_current_scheduled_energy:.4f} kWh"
)

print(
    f"Peak-period energy reduction: "
    f"{peak_energy_reduction_percent:.2f}%"
)

print(
    f"Average peak-period power reduction: "
    f"{average_peak_power_reduction_percent:.2f}%"
)


print("\nUSER / SAFETY")

print(
    f"Average comfort score: "
    f"{average_comfort:.2f}/100"
)

print(
    f"Safety overrides applied: "
    f"{int(test_df['safety_override'].sum()):,}"
)


print("\nFINAL ACTION COUNTS")

print(
    test_df[
        "final_action"
    ]
    .value_counts()
)


print("\nIMPORTANT INTERPRETATION:")

print(
    "These values are simulation-based PP1 estimates."
)

print(
    "They should not be presented as measured household "
    "electricity-bill savings."
)

print(
    "The tariff values are normalized experimental indices, "
    "and action effects are controlled simulation assumptions."
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
    "\nOperational benefit evaluation completed."
)