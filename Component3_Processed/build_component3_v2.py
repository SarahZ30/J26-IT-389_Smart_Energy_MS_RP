import pandas as pd
import numpy as np
import os


# ============================================================
# CONFIGURATION
# ============================================================

BASE_FILE = "output/component3_base_dataset.csv"
CONTEXT_FILE = "energydata_complete.csv"

OUTPUT_DIR = "v2_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

FINAL_FILE = os.path.join(
    OUTPUT_DIR,
    "component3_v2_dataset.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "component3_v2_label_summary.csv"
)

PROFILE_SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "component3_v2_profile_summary.csv"
)

ACTION_MATRIX_FILE = os.path.join(
    OUTPUT_DIR,
    "component3_v2_appliance_action_matrix.csv"
)

SAMPLE_FILE = os.path.join(
    OUTPUT_DIR,
    "component3_v2_sample.csv"
)

RANDOM_STATE = 42
TARGET_SIZE = 250000

rng = np.random.default_rng(RANDOM_STATE)


# ============================================================
# 1. LOAD BASE DATA
# ============================================================

print("============================================")
print("COMPONENT 3 - V2 DATASET CONSTRUCTION")
print("============================================")

print("\nLoading Component 3 base dataset...")

df = pd.read_csv(
    BASE_FILE,
    parse_dates=["Time"]
)

print(f"Base records: {len(df):,}")
print(f"Unique houses: {df['house_id'].nunique()}")


# ============================================================
# 2. SORT CHRONOLOGICALLY
# ============================================================

df = df.sort_values(
    [
        "house_id",
        "appliance_category",
        "Time"
    ]
).reset_index(drop=True)


# ============================================================
# 3. TEMPORAL / HISTORY FEATURES
# ============================================================

print("\nCreating temporal features...")

group_cols = [
    "house_id",
    "appliance_category"
]

grouped = df.groupby(
    group_cols,
    group_keys=False
)


# ------------------------------------------------------------
# Previous household-load observations
# ------------------------------------------------------------

df["load_lag_1"] = grouped[
    "aggregate_mean_w"
].shift(1)

df["load_lag_2"] = grouped[
    "aggregate_mean_w"
].shift(2)

df["load_lag_4"] = grouped[
    "aggregate_mean_w"
].shift(4)


# ------------------------------------------------------------
# Previous appliance state
# ------------------------------------------------------------

df["previous_appliance_active"] = grouped[
    "appliance_active"
].shift(1)


# ------------------------------------------------------------
# Rolling household demand
# 4 x 15 minutes = 1 hour
# ------------------------------------------------------------

df["rolling_load_mean_1h"] = (
    grouped["aggregate_mean_w"]
    .transform(
        lambda s:
        s.shift(1)
        .rolling(
            window=4,
            min_periods=1
        )
        .mean()
    )
)

df["rolling_load_max_1h"] = (
    grouped["aggregate_mean_w"]
    .transform(
        lambda s:
        s.shift(1)
        .rolling(
            window=4,
            min_periods=1
        )
        .max()
    )
)


# ============================================================
# 4. APPLIANCE STATE DURATION
# ============================================================

print("Creating appliance state-duration features...")

state_change = (
    df["appliance_active"]
    != grouped["appliance_active"].shift(1)
)

df["state_group"] = (
    state_change
    .groupby(
        [
            df["house_id"],
            df["appliance_category"]
        ]
    )
    .cumsum()
)

df["state_duration_steps"] = (
    df.groupby(
        [
            "house_id",
            "appliance_category",
            "state_group"
        ]
    )
    .cumcount()
    + 1
)

# Each record represents about 15 minutes
df["state_duration_minutes"] = (
    df["state_duration_steps"] * 15
)

df["on_duration_minutes"] = np.where(
    df["appliance_active"] == 1,
    df["state_duration_minutes"],
    0
)

df["off_duration_minutes"] = np.where(
    df["appliance_active"] == 0,
    df["state_duration_minutes"],
    0
)

df = df.drop(
    columns=[
        "state_group",
        "state_duration_steps"
    ]
)


# ============================================================
# 5. HISTORY-BASED LOAD FORECAST PROXY
# ============================================================

print("Creating history-based predicted load...")

# Temporary PP1 forecasting proxy.
# Unlike V1, this uses historical load rather than
# current load + random percentage.

df["predicted_load_w"] = (
    0.45 * df["load_lag_1"]
    + 0.20 * df["load_lag_2"]
    + 0.10 * df["load_lag_4"]
    + 0.25 * df["rolling_load_mean_1h"]
)

forecast_noise = rng.normal(
    loc=0,
    scale=0.10,
    size=len(df)
)

df["predicted_load_w"] = (
    df["predicted_load_w"]
    * (1 + forecast_noise)
)

df["predicted_load_w"] = (
    df["predicted_load_w"]
    .clip(lower=0)
)


# ============================================================
# 6. REMOVE ROWS WITHOUT ENOUGH HISTORY
# ============================================================

required_history = [
    "load_lag_1",
    "load_lag_2",
    "load_lag_4",
    "rolling_load_mean_1h",
    "rolling_load_max_1h",
    "previous_appliance_active",
    "predicted_load_w"
]

before = len(df)

df = df.dropna(
    subset=required_history
).copy()

print(
    f"Rows removed due to insufficient history: "
    f"{before - len(df):,}"
)


# ============================================================
# 7. PREDICTED LOAD LEVEL
# ============================================================

q33 = df[
    "predicted_load_w"
].quantile(0.33)

q66 = df[
    "predicted_load_w"
].quantile(0.66)


def load_level(value):

    if value <= q33:
        return "Low"

    elif value <= q66:
        return "Medium"

    return "High"


df["predicted_load_level"] = (
    df["predicted_load_w"]
    .apply(load_level)
)


# ============================================================
# 8. STRATIFIED RESEARCH SAMPLE
# ============================================================

print("\nCreating manageable research sample...")

df["sampling_group"] = (
    df["appliance_category"].astype(str)
    + "_"
    + df["appliance_active"].astype(str)
)

number_of_groups = df[
    "sampling_group"
].nunique()

per_group = max(
    1,
    TARGET_SIZE // number_of_groups
)

sample_parts = []

for name, group in df.groupby(
    "sampling_group"
):

    n = min(
        per_group,
        len(group)
    )

    sample_parts.append(
        group.sample(
            n=n,
            random_state=RANDOM_STATE
        )
    )

v2 = pd.concat(
    sample_parts,
    ignore_index=True
)

v2 = v2.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

v2 = v2.drop(
    columns=["sampling_group"]
)

print(
    f"V2 working records: {len(v2):,}"
)


# ============================================================
# 9. LOAD ENVIRONMENTAL DATASET
# ============================================================

print("\nLoading environmental dataset...")

context = pd.read_csv(
    CONTEXT_FILE
)

context["date"] = pd.to_datetime(
    context["date"],
    errors="coerce"
)

context = context.dropna(
    subset=["date"]
)

temperature_cols = [
    f"T{i}"
    for i in range(1, 10)
]

humidity_cols = [
    f"RH_{i}"
    for i in range(1, 10)
]

context["temperature_c"] = (
    context[
        temperature_cols
    ].mean(axis=1)
)

context["humidity_percent"] = (
    context[
        humidity_cols
    ].mean(axis=1)
)

context["context_hour"] = (
    context["date"].dt.hour
)


# ============================================================
# 10. ENVIRONMENTAL CONTEXT POOLS
# ============================================================

context_pools = {}

for hour in range(24):

    pool = context[
        context["context_hour"] == hour
    ][
        [
            "temperature_c",
            "humidity_percent",
            "T_out",
            "RH_out"
        ]
    ].dropna()

    context_pools[hour] = (
        pool.reset_index(drop=True)
    )


temperatures = []
humidities = []
outside_temps = []
outside_humidities = []

fallback_pool = context[
    [
        "temperature_c",
        "humidity_percent",
        "T_out",
        "RH_out"
    ]
].dropna().reset_index(drop=True)


for hour in v2["hour"]:

    pool = context_pools[
        int(hour)
    ]

    if len(pool) == 0:
        pool = fallback_pool

    idx = rng.integers(
        0,
        len(pool)
    )

    selected = pool.iloc[idx]

    temperatures.append(
        selected["temperature_c"]
    )

    humidities.append(
        selected["humidity_percent"]
    )

    outside_temps.append(
        selected["T_out"]
    )

    outside_humidities.append(
        selected["RH_out"]
    )


v2["temperature_c"] = temperatures

v2["humidity_percent"] = humidities

v2["outdoor_temperature_c"] = (
    outside_temps
)

v2["outdoor_humidity_percent"] = (
    outside_humidities
)


# ============================================================
# 11. PERSONALIZED USER PROFILES
# ============================================================

print("Creating personalized user profiles...")

profiles = [
    "Comfort_Oriented",
    "Balanced",
    "Energy_Saver",
    "Cost_Sensitive"
]

profile_probabilities = [
    0.20,
    0.40,
    0.20,
    0.20
]

v2["user_profile"] = rng.choice(
    profiles,
    size=len(v2),
    p=profile_probabilities
)


# ============================================================
# 12. USER PROFILE OBJECTIVE WEIGHTS
# ============================================================

PROFILE_WEIGHTS = {

    "Comfort_Oriented": {
        "comfort": 0.45,
        "cost": 0.15,
        "peak": 0.10,
        "preference": 0.30
    },

    "Balanced": {
        "comfort": 0.25,
        "cost": 0.25,
        "peak": 0.25,
        "preference": 0.25
    },

    "Energy_Saver": {
        "comfort": 0.15,
        "cost": 0.20,
        "peak": 0.40,
        "preference": 0.25
    },

    "Cost_Sensitive": {
        "comfort": 0.10,
        "cost": 0.45,
        "peak": 0.25,
        "preference": 0.20
    }
}


# ============================================================
# 13. USER FLEXIBILITY
# ============================================================

def generate_flexibility(row):

    appliance = row[
        "appliance_category"
    ]

    profile = row[
        "user_profile"
    ]

    # Refrigeration should have very little scheduling freedom
    if appliance == "Refrigeration":

        return rng.uniform(
            0.00,
            0.15
        )

    # Kettle is normally user-demand driven
    if appliance == "Kettle":

        return rng.uniform(
            0.05,
            0.35
        )

    # Comfort-oriented users accept fewer schedule changes
    if profile == "Comfort_Oriented":

        return rng.uniform(
            0.10,
            0.45
        )

    if profile == "Balanced":

        return rng.uniform(
            0.35,
            0.70
        )

    # Energy saver / cost-sensitive users
    return rng.uniform(
        0.60,
        1.00
    )


v2["user_flexibility"] = (
    v2.apply(
        generate_flexibility,
        axis=1
    )
)


# ============================================================
# 14. APPLIANCE OPERATING CONSTRAINTS
# ============================================================

def appliance_constraints(
    appliance
):

    if appliance == "Refrigeration":

        return pd.Series({

            "interruptible": 1,

            "max_delay_minutes": 15,

            "max_off_minutes": 45,

            "priority_level": 3

        })


    if appliance == "Washing Machine":

        return pd.Series({

            "interruptible": 0,

            "max_delay_minutes": 180,

            "max_off_minutes": 240,

            "priority_level": 1

        })


    if appliance == "Dishwasher":

        return pd.Series({

            "interruptible": 0,

            "max_delay_minutes": 240,

            "max_off_minutes": 300,

            "priority_level": 1

        })


    if appliance == "Kettle":

        return pd.Series({

            "interruptible": 1,

            "max_delay_minutes": 15,

            "max_off_minutes": 30,

            "priority_level": 2

        })


    return pd.Series({

        "interruptible": 1,

        "max_delay_minutes": 60,

        "max_off_minutes": 120,

        "priority_level": 1

    })


constraint_df = (
    v2["appliance_category"]
    .apply(
        appliance_constraints
    )
)

v2 = pd.concat(
    [
        v2,
        constraint_df
    ],
    axis=1
)


# ============================================================
# 15. OCCUPANCY SCENARIO
# ============================================================

def occupancy_probability(
    row
):

    hour = row["hour"]

    weekend = row["is_weekend"]

    if weekend == 1:

        if 7 <= hour <= 23:
            return 0.85

        return 0.55


    if 0 <= hour < 6:
        return 0.75


    if 6 <= hour < 9:
        return 0.90


    if 9 <= hour < 17:
        return 0.40


    if 17 <= hour <= 23:
        return 0.90


    return 0.50


occupancy_probs = v2.apply(
    occupancy_probability,
    axis=1
)

v2["occupancy"] = (
    rng.random(
        len(v2)
    )
    <
    occupancy_probs
).astype(int)


# ============================================================
# 16. TARIFF BAND
# ============================================================

# Temporary PP1 scheduling bands.
# Replace later with the finalized official tariff schedule.

def tariff_band(hour):

    if 0 <= hour < 6:

        return "Off_Peak"


    if 18 <= hour < 22:

        return "Peak"


    return "Day"


v2["tariff_band"] = (
    v2["hour"]
    .apply(
        tariff_band
    )
)


# ============================================================
# 17. TARIFF COST INDEX
# ============================================================

tariff_cost_index = {

    "Off_Peak": 0.30,

    "Day": 0.60,

    "Peak": 1.00

}

v2["tariff_cost_index"] = (
    v2["tariff_band"]
    .map(
        tariff_cost_index
    )
)


# ============================================================
# 18. NORMALIZED LOAD VARIABLES
# ============================================================

load_95 = v2[
    "predicted_load_w"
].quantile(0.95)

v2["normalized_predicted_load"] = (
    v2["predicted_load_w"]
    / max(load_95, 1)
).clip(
    0,
    1
)


power_95 = v2[
    "appliance_mean_w"
].quantile(0.95)

v2["normalized_appliance_power"] = (
    v2["appliance_mean_w"]
    / max(power_95, 1)
).clip(
    0,
    1
)


# ============================================================
# 19. POSSIBLE SCHEDULING ACTIONS
# ============================================================

ACTIONS = [

    "TURN_ON",

    "TURN_OFF",

    "DELAY",

    "ECO_MODE",

    "PRIORITY",

    "LOAD_BALANCE"

]


# ============================================================
# 20. IMPROVED MULTI-OBJECTIVE UTILITY FUNCTION
# ============================================================

def calculate_action_utility(
    row,
    action
):

    profile = row[
        "user_profile"
    ]

    weights = PROFILE_WEIGHTS[
        profile
    ]

    active = int(
        row["appliance_active"]
    )

    appliance = row[
        "appliance_category"
    ]

    load = float(
        row["normalized_predicted_load"]
    )

    cost = float(
        row["tariff_cost_index"]
    )

    flexibility = float(
        row["user_flexibility"]
    )

    occupancy = int(
        row["occupancy"]
    )

    priority = (
        float(row["priority_level"])
        / 3.0
    )

    max_off = max(
        float(
            row["max_off_minutes"]
        ),
        1.0
    )

    off_ratio = min(
        float(
            row["off_duration_minutes"]
        )
        / max_off,
        2.0
    )

    power = float(
        row["normalized_appliance_power"]
    )

    interruptible = int(
        row["interruptible"]
    )

    tariff = row[
        "tariff_band"
    ]

    predicted_level = row[
        "predicted_load_level"
    ]


    # --------------------------------------------------------
    # Default objective scores
    # --------------------------------------------------------

    comfort = 0.50

    cost_score = 0.50

    peak_score = 0.50

    preference_score = 0.50

    safety = 1.00


    # ========================================================
    # ACTION: TURN ON
    # ========================================================

    if action == "TURN_ON":

        comfort = (
            0.55
            + 0.25 * occupancy
            + 0.10 * (
                1 - flexibility
            )
        )

        cost_score = (
            0.75
            - 0.45 * cost
        )

        peak_score = (
            0.75
            - 0.50 * load
        )

        preference_score = (
            0.70
            - 0.30 * flexibility
        )


        if active == 1:

            comfort += 0.12


        if predicted_level == "High":

            peak_score -= 0.12


        if tariff == "Peak":

            cost_score -= 0.12


        if appliance == "Refrigeration":

            safety = 1.0


    # ========================================================
    # ACTION: TURN OFF
    # ========================================================

    elif action == "TURN_OFF":

        comfort = (
            0.68
            if active == 0
            else 0.18
        )

        cost_score = 0.92

        peak_score = (
            0.80
            + 0.15 * load
        )

        preference_score = (
            0.45
            + 0.30 * flexibility
        )


        # Avoid automatically making OFF dominant
        if active == 0:

            comfort += 0.05


        if active == 1:

            comfort -= 0.10


        if occupancy == 1:

            comfort -= 0.08


        # Refrigeration safety protection
        if appliance == "Refrigeration":

            if off_ratio >= 1.0:

                safety = 0.0

            elif off_ratio >= 0.70:

                safety = 0.20

            elif off_ratio >= 0.40:

                safety = 0.50


    # ========================================================
    # ACTION: DELAY
    # ========================================================

    elif action == "DELAY":

        # Only flexible appliances can realistically be delayed
        if appliance not in [
            "Washing Machine",
            "Dishwasher"
        ]:

            return -999.0


        comfort = (
            0.35
            + 0.45 * flexibility
        )

        cost_score = (
            0.55
            + 0.40 * cost
        )

        peak_score = (
            0.50
            + 0.45 * load
        )

        preference_score = (
            0.35
            + 0.60 * flexibility
        )


        # Peak periods favor delaying
        if tariff == "Peak":

            cost_score += 0.20


        # High predicted demand favors delaying
        if predicted_level == "High":

            peak_score += 0.20


        # Highly flexible user
        if flexibility >= 0.70:

            preference_score += 0.15


        # Delay is most logical before cycle starts
        if active == 0:

            comfort += 0.18

            preference_score += 0.12


        # Already-running non-interruptible appliance
        if (
            active == 1
            and interruptible == 0
        ):

            safety *= 0.12


        # Little value in delaying during off-peak periods
        if tariff == "Off_Peak":

            cost_score -= 0.22


    # ========================================================
    # ACTION: ECO MODE
    # ========================================================

    elif action == "ECO_MODE":

        comfort = (
            0.58
            + 0.10 * occupancy
        )

        cost_score = (
            0.62
            + 0.25 * cost
        )

        peak_score = (
            0.58
            + 0.25 * load
        )

        preference_score = (
            0.50
            + 0.30 * flexibility
        )


        if active == 1:

            comfort += 0.12

            cost_score += 0.07


        else:

            comfort -= 0.12


        if predicted_level == "Medium":

            peak_score += 0.12


        if tariff == "Peak":

            cost_score += 0.12


        if flexibility >= 0.60:

            preference_score += 0.08


        if appliance == "Refrigeration":

            safety *= 0.55


    # ========================================================
    # ACTION: PRIORITY
    # ========================================================

    elif action == "PRIORITY":

        comfort = (
            0.55
            + 0.25 * priority
        )

        cost_score = 0.25

        peak_score = 0.25

        preference_score = (
            0.50
            + 0.20 * priority
        )


        # Priority mainly makes sense for refrigeration
        if appliance == "Refrigeration":

            safety = 1.0

            comfort = (
                0.68
                + 0.28
                * min(
                    off_ratio,
                    1.0
                )
            )

            preference_score = (
                0.60
                + 0.22
                * min(
                    off_ratio,
                    1.0
                )
            )


            # Approaching safe OFF limit
            if off_ratio >= 1.0:

                comfort += 0.38

                preference_score += 0.28


            elif off_ratio >= 0.70:

                comfort += 0.27

                preference_score += 0.20


            elif off_ratio >= 0.50:

                comfort += 0.16

                preference_score += 0.12


            elif active == 1:

                comfort += 0.08


        else:

            # Non-critical loads should rarely receive priority
            safety *= 0.25


    # ========================================================
    # ACTION: LOAD BALANCE
    # ========================================================

    elif action == "LOAD_BALANCE":

        comfort = (
            0.48
            + 0.20 * flexibility
        )

        cost_score = (
            0.50
            + 0.25 * cost
        )

        peak_score = (
            0.45
            + 0.40 * load
            + 0.25 * power
        )

        preference_score = (
            0.50
            + 0.25 * flexibility
        )


        if predicted_level == "High":

            peak_score += 0.20


        if power >= 0.60:

            peak_score += 0.15


        if active == 1:

            peak_score += 0.10


        else:

            comfort -= 0.08


        # Kettle represents a short high-power balancing case
        if appliance == "Kettle":

            peak_score += 0.15


    # ========================================================
    # MULTI-OBJECTIVE TOTAL
    # ========================================================

    total = (

        weights["comfort"]
        * comfort

        +

        weights["cost"]
        * cost_score

        +

        weights["peak"]
        * peak_score

        +

        weights["preference"]
        * preference_score

    )


    # Apply safety factor
    total *= safety


    # ========================================================
    # PERSONALIZATION EFFECTS
    # ========================================================

    if profile == "Comfort_Oriented":

        if action in [
            "TURN_ON",
            "PRIORITY"
        ]:

            total += 0.06


        if action in [
            "DELAY",
            "TURN_OFF"
        ]:

            total -= 0.04


    elif profile == "Energy_Saver":

        if action in [
            "ECO_MODE",
            "LOAD_BALANCE",
            "DELAY"
        ]:

            total += 0.07


    elif profile == "Cost_Sensitive":

        if tariff == "Peak":

            if action in [
                "DELAY",
                "ECO_MODE",
                "TURN_OFF"
            ]:

                total += 0.08


    elif profile == "Balanced":

        if action in [
            "ECO_MODE",
            "LOAD_BALANCE"
        ]:

            total += 0.035


    # ========================================================
    # EXTRA SOFT CONTEXT BONUSES
    # ========================================================

    # Flexible appliances:
    # delay when expensive/high demand and user allows it
    if (
        action == "DELAY"
        and appliance in [
            "Washing Machine",
            "Dishwasher"
        ]
        and active == 0
        and flexibility >= 0.55
    ):

        total += 0.05


    if (
        action == "DELAY"
        and tariff == "Peak"
        and predicted_level == "High"
    ):

        total += 0.06


    # Refrigeration approaching maximum OFF duration
    if (
        action == "PRIORITY"
        and appliance == "Refrigeration"
        and off_ratio >= 0.50
    ):

        total += (
            0.10
            * min(
                off_ratio,
                1.0
            )
        )


    # High-load/high-power balancing situation
    if (
        action == "LOAD_BALANCE"
        and predicted_level == "High"
        and power >= 0.40
    ):

        total += 0.05


    # Eco operation under medium constraints
    if (
        action == "ECO_MODE"
        and predicted_level == "Medium"
        and active == 1
    ):

        total += 0.04


    # ========================================================
    # SMALL CONTROLLED UNCERTAINTY
    # ========================================================

    total += rng.normal(
        0,
        0.035
    )


    # ========================================================
    # STATE CONSISTENCY
    # ========================================================

    if (
        active == 0
        and action == "TURN_OFF"
    ):

        total += 0.025


    if (
        active == 1
        and action == "TURN_ON"
    ):

        total += 0.025


    return total


# ============================================================
# 21. CALCULATE UTILITY FOR EVERY ACTION
# ============================================================

print(
    "\nCalculating multi-objective scheduling utilities..."
)

all_action_scores = {}

for action in ACTIONS:

    print(
        f"Scoring action: {action}"
    )

    all_action_scores[action] = (
        v2.apply(
            lambda row:
            calculate_action_utility(
                row,
                action
            ),
            axis=1
        )
    )


score_df = pd.DataFrame(
    all_action_scores
)


# ============================================================
# 22. SELECT BEST ACTION
# ============================================================

v2["target_action"] = (
    score_df.idxmax(
        axis=1
    )
)

v2["winning_utility"] = (
    score_df.max(
        axis=1
    )
)


# ============================================================
# 23. DECISION MARGIN
# ============================================================

sorted_scores = np.sort(
    score_df.values,
    axis=1
)

v2["decision_margin"] = (
    sorted_scores[:, -1]
    - sorted_scores[:, -2]
)


# ============================================================
# 24. SAVE UTILITY VALUES FOR AUDIT PURPOSES
# ============================================================

for action in ACTIONS:

    v2[
        f"utility_{action}"
    ] = score_df[action]


# ============================================================
# 25. TARGET ACTION IDs
# ============================================================

ACTION_MAP = {

    "TURN_ON": 0,

    "TURN_OFF": 1,

    "DELAY": 2,

    "ECO_MODE": 3,

    "PRIORITY": 4,

    "LOAD_BALANCE": 5

}

v2["target_action_id"] = (
    v2["target_action"]
    .map(
        ACTION_MAP
    )
)


# ============================================================
# 26. VALIDATE ACTION GENERATION
# ============================================================

print("\nValidating generated scheduling classes...")

generated_actions = set(
    v2["target_action"]
    .unique()
)

expected_actions = set(
    ACTIONS
)

missing_actions = (
    expected_actions
    - generated_actions
)

if missing_actions:

    print(
        "\nWARNING:"
    )

    print(
        "The following actions were not generated:"
    )

    print(
        missing_actions
    )

else:

    print(
        "PASS: All six scheduling actions were generated."
    )


# ============================================================
# 27. DATA PROVENANCE
# ============================================================

v2["appliance_data_source"] = (
    "REFIT"
)

v2["environment_source"] = (
    "UCI_Appliances_Energy_Context_Augmentation"
)

v2["forecast_source"] = (
    "PP1_History_Based_Load_Forecast_Proxy"
)

v2["occupancy_source"] = (
    "PP1_Controlled_Scenario"
)

v2["user_profile_source"] = (
    "PP1_Personalization_Scenario"
)

v2["tariff_source"] = (
    "PP1_Temporary_Tariff_Bands"
)

v2["label_source"] = (
    "Multi_Objective_Utility_Policy_V2"
)


# ============================================================
# 28. SAVE FULL V2 DATASET
# ============================================================

print("\nSaving V2 dataset...")

v2.to_csv(
    FINAL_FILE,
    index=False
)


# ============================================================
# 29. LABEL SUMMARY
# ============================================================

label_summary = (
    v2.groupby(
        "target_action"
    )
    .agg(

        records=(
            "target_action",
            "size"
        ),

        houses=(
            "house_id",
            "nunique"
        ),

        avg_utility=(
            "winning_utility",
            "mean"
        ),

        avg_decision_margin=(
            "decision_margin",
            "mean"
        )

    )
    .reset_index()
)

label_summary["percentage"] = (
    label_summary["records"]
    / len(v2)
    * 100
)

label_summary = (
    label_summary
    .sort_values(
        "records",
        ascending=False
    )
    .reset_index(drop=True)
)

label_summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# 30. USER PROFILE SUMMARY
# ============================================================

profile_summary = pd.crosstab(
    v2["user_profile"],
    v2["target_action"]
)

profile_summary.to_csv(
    PROFILE_SUMMARY_FILE
)


# ============================================================
# 31. APPLIANCE x ACTION MATRIX
# ============================================================

action_matrix = pd.crosstab(
    v2["appliance_category"],
    v2["target_action"]
)

action_matrix.to_csv(
    ACTION_MATRIX_FILE
)


# ============================================================
# 32. CREATE SAMPLE DATASET
# ============================================================

sample_parts = []

for action, group in v2.groupby(
    "target_action"
):

    n = min(
        750,
        len(group)
    )

    sample_parts.append(
        group.sample(
            n=n,
            random_state=RANDOM_STATE
        )
    )

sample = pd.concat(
    sample_parts,
    ignore_index=True
)

sample = sample.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

sample.to_csv(
    SAMPLE_FILE,
    index=False
)


# ============================================================
# 33. AMBIGUOUS DECISION SUMMARY
# ============================================================

ambiguous_threshold = 0.05

ambiguous_count = (
    v2[
        "decision_margin"
    ]
    < ambiguous_threshold
).sum()

ambiguous_percent = (
    ambiguous_count
    / len(v2)
    * 100
)


# ============================================================
# 34. FINAL REPORT
# ============================================================

print("\n============================================")
print("V2 DATASET CONSTRUCTION COMPLETED")
print("============================================")

print(
    f"\nTotal V2 records: "
    f"{len(v2):,}"
)

print("\nTarget distribution:")

print(
    v2["target_action"]
    .value_counts()
)

print("\nTarget percentages:")

print(
    (
        v2["target_action"]
        .value_counts(
            normalize=True
        )
        * 100
    ).round(2)
)


print("\nUser profiles:")

print(
    v2["user_profile"]
    .value_counts()
)


print("\nAppliance x Action Matrix:")

print(
    action_matrix
)


print("\nDecision margin statistics:")

print(
    v2["decision_margin"]
    .describe()
)


print(
    f"\nAmbiguous decisions "
    f"(margin < {ambiguous_threshold}): "
    f"{ambiguous_count:,}"
)

print(
    f"Ambiguous percentage: "
    f"{ambiguous_percent:.2f}%"
)


print("\nGenerated classes:")

print(
    sorted(
        generated_actions
    )
)


print("\nGenerated files:")

print(
    FINAL_FILE
)

print(
    SUMMARY_FILE
)

print(
    PROFILE_SUMMARY_FILE
)

print(
    ACTION_MATRIX_FILE
)

print(
    SAMPLE_FILE
)

print("\nDone.")