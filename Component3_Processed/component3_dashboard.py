import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Component 3 - Smart Appliance Scheduler",
    page_icon="⚡",
    layout="wide"
)


# ============================================================
# FILE PATHS
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

IMPORTANCE_FILE = (
    "explainability_output/results/"
    "raw_feature_group_importance.csv"
)


# ============================================================
# CONFIDENCE-GATE CONFIGURATION
# ============================================================

# Experimental PP1 threshold selected from the confidence
# evaluation results:
#
# Low confidence (< 0.60) showed approximately 53% accuracy.
#
# Therefore predictions below 60% are not permitted to
# directly change the appliance state.

CONFIDENCE_THRESHOLD = 0.60


# ============================================================
# RESOURCE VALIDATION
# ============================================================

required_files = [
    DATA_FILE,
    MODEL_FILE,
    LABEL_ENCODER_FILE
]

missing_files = [
    path
    for path in required_files
    if not os.path.exists(path)
]

if missing_files:

    st.error(
        "Required project files are missing."
    )

    for path in missing_files:

        st.write(
            f"- {path}"
        )

    st.stop()


# ============================================================
# LOAD RESOURCES
# ============================================================

@st.cache_resource
def load_model():

    pipeline = joblib.load(
        MODEL_FILE
    )

    encoder = joblib.load(
        LABEL_ENCODER_FILE
    )

    return pipeline, encoder


@st.cache_data
def load_dataset():

    return pd.read_csv(
        DATA_FILE
    )


@st.cache_data
def load_importance():

    if os.path.exists(
        IMPORTANCE_FILE
    ):

        return pd.read_csv(
            IMPORTANCE_FILE
        )

    return None


try:

    pipeline, label_encoder = load_model()

    df = load_dataset()

    importance_df = load_importance()

except Exception as error:

    st.error(
        "Failed to load Component 3 resources."
    )

    st.exception(
        error
    )

    st.stop()


# ============================================================
# EXTRACT MODEL COMPONENTS
# ============================================================

try:

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    classifier = pipeline.named_steps[
        "classifier"
    ]

except Exception as error:

    st.error(
        "The saved model pipeline does not contain the "
        "expected preprocessor and classifier stages."
    )

    st.exception(
        error
    )

    st.stop()


# ============================================================
# EXTRACT EXACT MODEL FEATURES
# ============================================================

categorical_features = []
numeric_features = []


for (
    transformer_name,
    transformer,
    columns
) in preprocessor.transformers_:

    if transformer_name == "categorical":

        categorical_features = list(
            columns
        )

    elif transformer_name == "numeric":

        numeric_features = list(
            columns
        )


required_features = (
    categorical_features
    + numeric_features
)


if len(
    required_features
) == 0:

    st.error(
        "No model features were found in the saved pipeline."
    )

    st.stop()


# ============================================================
# SAFE CONVERSION HELPERS
# ============================================================

def safe_float(
    value,
    default=0.0
):

    try:

        if pd.isna(
            value
        ):

            return float(
                default
            )

        return float(
            value
        )

    except Exception:

        return float(
            default
        )


def safe_int(
    value,
    default=0
):

    try:

        if pd.isna(
            value
        ):

            return int(
                default
            )

        return int(
            value
        )

    except Exception:

        return int(
            default
        )


# ============================================================
# APPLIANCE OPERATING CONSTRAINTS
# ============================================================

APPLIANCE_RULES = {

    "Refrigeration": {

        "critical_flag": 1,
        "flexible_flag": 0,
        "high_power_short_use": 0,
        "interruptible": 1,
        "priority_level": 3,
        "max_delay_minutes": 15,
        "max_off_minutes": 45
    },

    "Washing Machine": {

        "critical_flag": 0,
        "flexible_flag": 1,
        "high_power_short_use": 0,
        "interruptible": 0,
        "priority_level": 1,
        "max_delay_minutes": 180,
        "max_off_minutes": 240
    },

    "Dishwasher": {

        "critical_flag": 0,
        "flexible_flag": 1,
        "high_power_short_use": 0,
        "interruptible": 0,
        "priority_level": 1,
        "max_delay_minutes": 240,
        "max_off_minutes": 300
    },

    "Kettle": {

        "critical_flag": 0,
        "flexible_flag": 0,
        "high_power_short_use": 1,
        "interruptible": 1,
        "priority_level": 2,
        "max_delay_minutes": 15,
        "max_off_minutes": 30
    }
}


# ============================================================
# USER PROFILE DEFAULT FLEXIBILITY
# ============================================================

PROFILE_FLEXIBILITY = {

    "Comfort_Oriented": 0.20,
    "Balanced": 0.50,
    "Energy_Saver": 0.80,
    "Cost_Sensitive": 0.90
}


# ============================================================
# EXPERIMENTAL NORMALIZED TARIFF INDEX
# ============================================================

TARIFF_COSTS = {

    "Off_Peak": 0.30,
    "Day": 0.60,
    "Peak": 1.00
}


# ============================================================
# RELAY MAPPING
# ============================================================

RELAY_MAPPING = {

    "Refrigeration":
        "Relay Channel 1",

    "Washing Machine":
        "Relay Channel 2",

    "Dishwasher":
        "Relay Channel 3",

    "Kettle":
        "Relay Channel 4"
}


# ============================================================
# CONFIDENCE GATE
# ============================================================

def apply_confidence_gate(
    row,
    ml_action,
    confidence
):

    ml_action = str(
        ml_action
    )

    confidence = float(
        confidence
    )

    appliance_active = safe_int(
        row[
            "appliance_active"
        ]
    )


    # --------------------------------------------------------
    # Model confidence is sufficient
    # --------------------------------------------------------

    if confidence >= CONFIDENCE_THRESHOLD:

        return (
            ml_action,
            False,
            "CONFIDENCE_ACCEPTED"
        )


    # --------------------------------------------------------
    # Low confidence
    #
    # Preserve current operating state rather than allowing
    # an uncertain ML recommendation to directly actuate
    # hardware.
    # --------------------------------------------------------

    if appliance_active == 1:

        fallback_action = (
            "TURN_ON"
        )

    else:

        fallback_action = (
            "TURN_OFF"
        )


    return (
        fallback_action,
        True,
        "LOW_CONFIDENCE_HOLD_CURRENT_STATE"
    )


# ============================================================
# SAFETY RULE IDENTIFICATION
# ============================================================

def identify_safety_violation(
    row,
    action
):

    action = str(
        action
    )

    appliance = str(
        row[
            "appliance_category"
        ]
    )

    active = safe_int(
        row[
            "appliance_active"
        ]
    )

    interruptible = safe_int(
        row[
            "interruptible"
        ]
    )

    off_duration = safe_float(
        row[
            "off_duration_minutes"
        ]
    )

    max_off = max(
        safe_float(
            row[
                "max_off_minutes"
            ],
            1.0
        ),
        1.0
    )

    max_delay = max(
        safe_float(
            row[
                "max_delay_minutes"
            ],
            1.0
        ),
        1.0
    )


    # ========================================================
    # S1 - Refrigerator maximum OFF duration exceeded
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
    # S2 - Refrigeration should not receive DELAY
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
    # S3 - Non-interruptible active cycle protection
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
    # S4 - Maximum delay exceeded
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
# SAFETY-LAYER OVERRIDE
# ============================================================

def apply_safety_layer(
    row,
    candidate_action
):

    candidate_action = str(
        candidate_action
    )

    violation, rule = (
        identify_safety_violation(
            row,
            candidate_action
        )
    )


    if not violation:

        return (
            candidate_action,
            False,
            "NONE"
        )


    if rule in [
        "S1_CRITICAL_MAX_OFF_EXCEEDED",
        "S2_CRITICAL_APPLIANCE_DELAY"
    ]:

        safe_action = (
            "PRIORITY"
        )


    elif rule in [
        "S3_NON_INTERRUPTIBLE_CYCLE",
        "S4_MAX_DELAY_EXCEEDED"
    ]:

        safe_action = (
            "TURN_ON"
        )


    else:

        safe_action = (
            candidate_action
        )


    return (
        safe_action,
        safe_action != candidate_action,
        rule
    )


# ============================================================
# TEMPLATE RECORD
# ============================================================

def get_template_row(
    appliance
):

    appliance_rows = df[
        df[
            "appliance_category"
        ] == appliance
    ]


    if len(
        appliance_rows
    ) == 0:

        template = (
            df.iloc[
                [0]
            ]
            .copy()
        )

    else:

        template = (
            appliance_rows.sample(
                n=1,
                random_state=42
            )
            .copy()
        )


    return (
        template
        .reset_index(
            drop=True
        )
    )


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def calculate_normalized_predicted_load(
    predicted_load
):

    if (
        "predicted_load_w"
        not in df.columns
    ):

        return float(
            np.clip(
                predicted_load / 5000.0,
                0.0,
                1.0
            )
        )


    upper_limit = safe_float(
        df[
            "predicted_load_w"
        ].quantile(
            0.99
        ),
        5000.0
    )


    upper_limit = max(
        upper_limit,
        1.0
    )


    return float(
        np.clip(
            predicted_load
            / upper_limit,
            0.0,
            1.0
        )
    )


def calculate_normalized_appliance_power(
    appliance,
    appliance_power
):

    category_df = df[
        df[
            "appliance_category"
        ] == appliance
    ]


    if (
        len(
            category_df
        ) == 0
        or "appliance_mean_w"
        not in category_df.columns
    ):

        upper_limit = 3000.0

    else:

        upper_limit = safe_float(

            category_df[
                "appliance_mean_w"
            ].quantile(
                0.99
            ),

            3000.0
        )


    upper_limit = max(
        upper_limit,
        1.0
    )


    return float(
        np.clip(
            appliance_power
            / upper_limit,
            0.0,
            1.0
        )
    )


# ============================================================
# LOAD LEVEL
# ============================================================

def get_load_level(
    normalized_value
):

    if normalized_value < 0.33:

        return "Low"

    elif normalized_value < 0.66:

        return "Medium"

    else:

        return "High"


# ============================================================
# TIME PERIOD
# ============================================================

def get_time_period(
    hour
):

    if hour < 6:

        return "Night"

    elif hour < 12:

        return "Morning"

    elif hour < 18:

        return "Afternoon"

    else:

        return "Evening"


# ============================================================
# MQTT PAYLOAD BUILDER
# ============================================================

def create_mqtt_payload(
    appliance,
    ml_action,
    gated_action,
    final_action,
    confidence,
    confidence_gate_triggered,
    confidence_gate_reason,
    safety_override,
    safety_rule
):

    return {

        "component":
            "component_3",

        "appliance":
            appliance,

        "ml_recommendation":
            ml_action,

        "confidence":
            round(
                float(
                    confidence
                ),
                4
            ),

        "confidence_threshold":
            CONFIDENCE_THRESHOLD,

        "confidence_gate_triggered":
            bool(
                confidence_gate_triggered
            ),

        "confidence_gate_reason":
            confidence_gate_reason,

        "gated_action":
            gated_action,

        "safety_override":
            bool(
                safety_override
            ),

        "safety_rule":
            safety_rule,

        "final_action":
            final_action
    }


# ============================================================
# DECISION CONTEXT EXPLANATION
# ============================================================

def create_decision_context(
    row,
    final_action
):

    reasons = []


    appliance = str(
        row[
            "appliance_category"
        ]
    )

    profile = str(
        row[
            "user_profile"
        ]
    )

    tariff = str(
        row[
            "tariff_band"
        ]
    )

    load_level = str(
        row[
            "predicted_load_level"
        ]
    )

    flexibility = safe_float(
        row[
            "user_flexibility"
        ]
    )

    occupancy = safe_int(
        row[
            "occupancy"
        ]
    )


    # --------------------------------------------------------
    # Tariff
    # --------------------------------------------------------

    if tariff == "Peak":

        reasons.append(
            "Peak tariff increases the incentive to reduce "
            "or shift appliance demand."
        )

    elif tariff == "Off_Peak":

        reasons.append(
            "Off-peak tariff reduces the incentive to "
            "postpone appliance operation."
        )


    # --------------------------------------------------------
    # Predicted demand
    # --------------------------------------------------------

    if load_level == "High":

        reasons.append(
            "Predicted household demand is high, increasing "
            "load-management pressure."
        )

    elif load_level == "Low":

        reasons.append(
            "Predicted household demand is relatively low."
        )


    # --------------------------------------------------------
    # User profile
    # --------------------------------------------------------

    if profile == "Comfort_Oriented":

        reasons.append(
            "The Comfort-Oriented profile places greater "
            "importance on immediate appliance availability."
        )

    elif profile == "Balanced":

        reasons.append(
            "The Balanced profile considers comfort, energy, "
            "tariff cost, and scheduling flexibility."
        )

    elif profile == "Energy_Saver":

        reasons.append(
            "The Energy-Saver profile accepts stronger "
            "energy-management intervention."
        )

    elif profile == "Cost_Sensitive":

        reasons.append(
            "The Cost-Sensitive profile gives greater "
            "importance to tariff-related savings."
        )


    # --------------------------------------------------------
    # User flexibility
    # --------------------------------------------------------

    if flexibility >= 0.70:

        reasons.append(
            "High user flexibility permits stronger "
            "scheduling adjustments."
        )

    elif flexibility <= 0.30:

        reasons.append(
            "Low user flexibility discourages aggressive "
            "scheduling changes."
        )


    # --------------------------------------------------------
    # Appliance characteristics
    # --------------------------------------------------------

    if appliance == "Refrigeration":

        reasons.append(
            "Refrigeration is treated as a critical "
            "high-priority appliance."
        )

    elif appliance in [
        "Washing Machine",
        "Dishwasher"
    ]:

        reasons.append(
            "The appliance is treated as a flexible but "
            "non-interruptible cycle-based load."
        )

    elif appliance == "Kettle":

        reasons.append(
            "The kettle is treated as a short-duration "
            "high-power appliance."
        )


    # --------------------------------------------------------
    # Occupancy
    # --------------------------------------------------------

    if occupancy == 1:

        reasons.append(
            "The home is occupied, so immediate user "
            "comfort may have greater relevance."
        )


    # --------------------------------------------------------
    # Final action
    # --------------------------------------------------------

    if final_action == "DELAY":

        reasons.append(
            "The final controller action postpones appliance "
            "operation to a more suitable period."
        )

    elif final_action == "ECO_MODE":

        reasons.append(
            "The final controller action reduces energy "
            "demand while maintaining appliance operation."
        )

    elif final_action == "LOAD_BALANCE":

        reasons.append(
            "The final controller action reduces demand "
            "concentration during the current interval."
        )

    elif final_action == "PRIORITY":

        reasons.append(
            "The appliance receives priority because "
            "operational constraints outweigh optimization."
        )

    elif final_action == "TURN_ON":

        reasons.append(
            "The final controller action maintains or enables "
            "appliance operation."
        )

    elif final_action == "TURN_OFF":

        reasons.append(
            "The final controller action keeps or switches "
            "the appliance off under the current context."
        )


    return reasons


# ============================================================
# HEADER
# ============================================================

st.title(
    "⚡ Personalized Smart Appliance Scheduling"
)

st.caption(
    "Component 3 — AI-Based Smart Energy Management System"
)

st.info(
    "This proof-of-concept integrates the trained XGBoost "
    "personalized scheduler, prediction-confidence gating, "
    "rule-based safety validation, explainability, and "
    "MQTT-ready command generation."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "System Information"
    )

    st.write(
        "**Selected Model:** XGBoost"
    )

    st.write(
        f"**Raw Model Features:** {len(required_features)}"
    )

    st.write(
        "**Scheduling Actions:** 6"
    )

    st.write(
        f"**Confidence Gate:** "
        f"{CONFIDENCE_THRESHOLD * 100:.0f}%"
    )

    st.write(
        "**Safety Rules:** 4"
    )

    st.write(
        "**Test Accuracy:** 84.7%"
    )

    st.write(
        "**Macro F1:** 84.3%"
    )

    st.write(
        "**Safety Compliance:** 100%"
    )

    st.divider()

    st.caption(
        "PP1 research proof-of-concept. Operational energy "
        "and tariff values are simulation-based estimates."
    )


# ============================================================
# INPUT SECTION
# ============================================================

st.header(
    "1. Appliance & User Context"
)


input_col1, input_col2, input_col3 = st.columns(
    3
)


with input_col1:

    appliance = st.selectbox(

        "Appliance",

        options=[
            "Refrigeration",
            "Washing Machine",
            "Dishwasher",
            "Kettle"
        ]
    )


with input_col2:

    user_profile = st.selectbox(

        "User Profile",

        options=[
            "Balanced",
            "Comfort_Oriented",
            "Energy_Saver",
            "Cost_Sensitive"
        ]
    )


with input_col3:

    tariff_band = st.selectbox(

        "Tariff Period",

        options=[
            "Off_Peak",
            "Day",
            "Peak"
        ],

        index=1
    )


# ============================================================
# CREATE TEMPLATE RECORD
# ============================================================

input_df = get_template_row(
    appliance
)


# ============================================================
# APPLY APPLIANCE CONSTRAINTS
# ============================================================

appliance_rules = (
    APPLIANCE_RULES[
        appliance
    ]
)


for (
    column,
    value
) in appliance_rules.items():

    if column in input_df.columns:

        input_df.loc[
            0,
            column
        ] = value


# ============================================================
# SECOND INPUT ROW
# ============================================================

input_col4, input_col5, input_col6 = st.columns(
    3
)


with input_col4:

    appliance_active = st.toggle(
        "Appliance Currently ON",
        value=True
    )


with input_col5:

    occupancy = st.toggle(
        "Home Occupied",
        value=True
    )


with input_col6:

    user_flexibility = st.slider(

        "User Flexibility",

        min_value=0.0,
        max_value=1.0,

        value=float(
            PROFILE_FLEXIBILITY[
                user_profile
            ]
        ),

        step=0.05
    )


# ============================================================
# DEFAULT VALUES
# ============================================================

template_power = 0.0

if "appliance_mean_w" in input_df.columns:

    template_power = safe_float(
        input_df.loc[
            0,
            "appliance_mean_w"
        ]
    )


template_power = float(
    np.clip(
        template_power,
        0.0,
        5000.0
    )
)


template_predicted_load = 500.0

if "predicted_load_w" in input_df.columns:

    template_predicted_load = safe_float(
        input_df.loc[
            0,
            "predicted_load_w"
        ],
        500.0
    )


template_predicted_load = float(
    np.clip(
        template_predicted_load,
        0.0,
        10000.0
    )
)


template_off_duration = 0.0

if "off_duration_minutes" in input_df.columns:

    template_off_duration = safe_float(
        input_df.loc[
            0,
            "off_duration_minutes"
        ]
    )


template_off_duration = float(
    np.clip(
        template_off_duration,
        0.0,
        1000.0
    )
)


# ============================================================
# THIRD INPUT ROW
# ============================================================

input_col7, input_col8, input_col9 = st.columns(
    3
)


with input_col7:

    appliance_power = st.number_input(

        "Current Appliance Power (W)",

        min_value=0.0,
        max_value=5000.0,

        value=template_power,

        step=10.0
    )


with input_col8:

    predicted_load = st.number_input(

        "Predicted Household Load (W)",

        min_value=0.0,
        max_value=10000.0,

        value=template_predicted_load,

        step=50.0
    )


with input_col9:

    off_duration = st.number_input(

        "Current OFF Duration (Minutes)",

        min_value=0.0,
        max_value=1000.0,

        value=template_off_duration,

        step=15.0
    )


# ============================================================
# ADVANCED INPUTS
# ============================================================

with st.expander(
    "Environmental & Advanced Inputs"
):

    env_col1, env_col2, env_col3 = st.columns(
        3
    )


    with env_col1:

        indoor_temp = st.number_input(

            "Indoor Temperature (°C)",

            min_value=10.0,
            max_value=40.0,

            value=24.0,

            step=0.5
        )


    with env_col2:

        humidity = st.number_input(

            "Indoor Relative Humidity (%)",

            min_value=10.0,
            max_value=100.0,

            value=50.0,

            step=1.0
        )


    with env_col3:

        hour = st.slider(

            "Hour of Day",

            min_value=0,
            max_value=23,

            value=18
        )


# ============================================================
# UPDATE INPUT DATAFRAME
# ============================================================

base_updates = {

    "appliance_category":
        appliance,

    "user_profile":
        user_profile,

    "user_flexibility":
        float(
            user_flexibility
        ),

    "tariff_band":
        tariff_band,

    "tariff_cost_index":
        float(
            TARIFF_COSTS[
                tariff_band
            ]
        ),

    "appliance_active":
        int(
            appliance_active
        ),

    "occupancy":
        int(
            occupancy
        ),

    "appliance_mean_w":
        float(
            appliance_power
        ),

    "predicted_load_w":
        float(
            predicted_load
        ),

    "off_duration_minutes":
        float(
            off_duration
        ),

    "hour":
        int(
            hour
        )
}


for (
    column,
    value
) in base_updates.items():

    if column in input_df.columns:

        input_df.loc[
            0,
            column
        ] = value


# ============================================================
# RELATED STATE FEATURES
# ============================================================

if "previous_appliance_active" in input_df.columns:

    input_df.loc[
        0,
        "previous_appliance_active"
    ] = int(
        appliance_active
    )


if "on_duration_minutes" in input_df.columns:

    input_df.loc[
        0,
        "on_duration_minutes"
    ] = (
        30.0
        if appliance_active
        else 0.0
    )


if "state_duration_minutes" in input_df.columns:

    input_df.loc[
        0,
        "state_duration_minutes"
    ] = (
        30.0
        if appliance_active
        else float(
            off_duration
        )
    )


# ============================================================
# ENVIRONMENT FEATURES
# ============================================================

temperature_candidates = [
    "indoor_temperature",
    "temperature",
    "T1"
]


for column in temperature_candidates:

    if column in input_df.columns:

        input_df.loc[
            0,
            column
        ] = float(
            indoor_temp
        )


humidity_candidates = [
    "indoor_humidity",
    "humidity",
    "RH_1",
    "RH1"
]


for column in humidity_candidates:

    if column in input_df.columns:

        input_df.loc[
            0,
            column
        ] = float(
            humidity
        )


# ============================================================
# TIME PERIOD
# ============================================================

calculated_period = get_time_period(
    hour
)


if "time_period" in input_df.columns:

    known_periods = (
        df[
            "time_period"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


    if calculated_period in known_periods:

        input_df.loc[
            0,
            "time_period"
        ] = calculated_period


# ============================================================
# NORMALIZED PREDICTED LOAD
# ============================================================

normalized_predicted_load = (
    calculate_normalized_predicted_load(
        predicted_load
    )
)


if "normalized_predicted_load" in input_df.columns:

    input_df.loc[
        0,
        "normalized_predicted_load"
    ] = normalized_predicted_load


# ============================================================
# PREDICTED LOAD LEVEL
# ============================================================

calculated_load_level = (
    get_load_level(
        normalized_predicted_load
    )
)


if "predicted_load_level" in input_df.columns:

    known_levels = (
        df[
            "predicted_load_level"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


    if calculated_load_level in known_levels:

        input_df.loc[
            0,
            "predicted_load_level"
        ] = calculated_load_level


# ============================================================
# NORMALIZED APPLIANCE POWER
# ============================================================

normalized_appliance_power = (
    calculate_normalized_appliance_power(
        appliance,
        appliance_power
    )
)


if "normalized_appliance_power" in input_df.columns:

    input_df.loc[
        0,
        "normalized_appliance_power"
    ] = normalized_appliance_power


# ============================================================
# REAPPLY FIXED APPLIANCE CONSTRAINTS
# ============================================================

for (
    column,
    value
) in appliance_rules.items():

    if column in input_df.columns:

        input_df.loc[
            0,
            column
        ] = value


# ============================================================
# MODEL INPUT VALIDATION
# ============================================================

missing_model_features = [

    feature
    for feature in required_features
    if feature not in input_df.columns
]


if missing_model_features:

    st.error(
        "The dashboard could not create all required "
        "model input features."
    )

    st.write(
        missing_model_features
    )

    st.stop()


X_input = input_df[
    required_features
].copy()


# ============================================================
# AI DECISION SECTION
# ============================================================

st.divider()

st.header(
    "2. AI Scheduling Decision"
)


run_prediction = st.button(

    "⚡ Generate Scheduling Decision",

    type="primary",

    width="stretch"
)


if run_prediction:

    try:

        # ====================================================
        # XGBOOST PREDICTION
        # ====================================================

        prediction_array = (
            pipeline.predict(
                X_input
            )
        )


        prediction_id = int(
            prediction_array[
                0
            ]
        )


        probability_array = (
            pipeline.predict_proba(
                X_input
            )[0]
        )


        ml_action = str(
            label_encoder.inverse_transform(
                [
                    prediction_id
                ]
            )[0]
        )


        confidence = float(
            probability_array[
                prediction_id
            ]
        )


        # ====================================================
        # CONFIDENCE GATE
        # ====================================================

        (
            gated_action,
            confidence_gate_triggered,
            confidence_gate_reason
        ) = apply_confidence_gate(

            input_df.iloc[
                0
            ],

            ml_action,

            confidence
        )


        # ====================================================
        # SAFETY LAYER
        # ====================================================

        (
            final_action,
            safety_override,
            safety_rule
        ) = apply_safety_layer(

            input_df.iloc[
                0
            ],

            gated_action
        )


        # ====================================================
        # RESULT CARDS
        # ====================================================

        (
            result_col1,
            result_col2,
            result_col3,
            result_col4
        ) = st.columns(
            4
        )


        with result_col1:

            st.metric(
                "ML Recommendation",
                ml_action
            )


        with result_col2:

            st.metric(
                "Prediction Confidence",
                f"{confidence * 100:.2f}%"
            )


        with result_col3:

            st.metric(
                "Confidence-Gated Action",
                gated_action
            )


        with result_col4:

            st.metric(
                "Final Safe Action",
                final_action
            )


        # ====================================================
        # CONFIDENCE VALIDATION
        # ====================================================

        st.subheader(
            "Confidence Validation"
        )


        if confidence >= 0.90:

            st.success(
                "Very-high confidence. The ML recommendation "
                "was accepted by the confidence gate."
            )


        elif confidence >= 0.75:

            st.success(
                "High confidence. The ML recommendation "
                "was accepted by the confidence gate."
            )


        elif confidence >= CONFIDENCE_THRESHOLD:

            st.info(
                "Medium confidence. The prediction exceeds "
                "the minimum autonomous-control threshold "
                "and was accepted."
            )


        else:

            st.warning(
                "Low-confidence prediction detected. "
                "The ML recommendation was not directly "
                "approved for autonomous actuation."
            )


            confidence_col1, confidence_col2 = (
                st.columns(
                    2
                )
            )


            with confidence_col1:

                st.write(
                    "**Original ML Recommendation**"
                )

                st.code(
                    ml_action
                )

                st.write(
                    "**Prediction Confidence**"
                )

                st.code(
                    f"{confidence * 100:.2f}%"
                )


            with confidence_col2:

                st.write(
                    "**Required Confidence Threshold**"
                )

                st.code(
                    f"{CONFIDENCE_THRESHOLD * 100:.0f}%"
                )

                st.write(
                    "**Conservative Fallback**"
                )

                st.code(
                    gated_action
                )


            st.caption(
                "The fallback preserves the current appliance "
                "state before deterministic safety validation."
            )


        # ====================================================
        # SAFETY VALIDATION
        # ====================================================

        st.subheader(
            "Safety Validation"
        )


        if safety_override:

            st.error(
                "Rule-based safety override activated."
            )


            safety_col1, safety_col2 = (
                st.columns(
                    2
                )
            )


            with safety_col1:

                st.write(
                    "**Action Entering Safety Layer**"
                )

                st.code(
                    gated_action
                )


            with safety_col2:

                st.write(
                    "**Final Safe Action**"
                )

                st.code(
                    final_action
                )


            st.write(
                "**Activated Safety Rule**"
            )

            st.code(
                safety_rule
            )


        else:

            st.success(
                "No safety constraint violation detected. "
                "The confidence-gated action passed the "
                "rule-based safety layer."
            )


        # ====================================================
        # DECISION PIPELINE SUMMARY
        # ====================================================

        st.subheader(
            "Decision Pipeline"
        )


        safety_stage_result = (
            safety_rule
            if safety_override
            else "PASSED"
        )


        decision_pipeline_df = pd.DataFrame({

            "Stage": [

                "Machine Learning",
                "Confidence Gate",
                "Safety Layer",
                "Final Controller Output"
            ],

            "Result": [

                str(
                    ml_action
                ),

                str(
                    gated_action
                ),

                str(
                    safety_stage_result
                ),

                str(
                    final_action
                )
            ]
        })


        decision_pipeline_df[
            "Stage"
        ] = decision_pipeline_df[
            "Stage"
        ].astype(
            str
        )


        decision_pipeline_df[
            "Result"
        ] = decision_pipeline_df[
            "Result"
        ].astype(
            str
        )


        st.dataframe(
            decision_pipeline_df,
            width="stretch",
            hide_index=True
        )


        # ====================================================
        # PREDICTION PROBABILITIES
        # ====================================================

        st.subheader(
            "Prediction Probabilities"
        )


        probability_df = pd.DataFrame({

            "Action":
                [
                    str(
                        value
                    )
                    for value
                    in label_encoder.classes_
                ],

            "Probability":
                [
                    float(
                        value
                    )
                    for value
                    in probability_array
                ]
        })


        probability_df[
            "Probability (%)"
        ] = (
            probability_df[
                "Probability"
            ]
            * 100.0
        )


        probability_df = (
            probability_df
            .sort_values(
                "Probability",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )


        st.dataframe(

            probability_df[
                [
                    "Action",
                    "Probability (%)"
                ]
            ],

            width="stretch",

            hide_index=True
        )


        probability_chart_df = (
            probability_df[
                [
                    "Action",
                    "Probability"
                ]
            ]
            .set_index(
                "Action"
            )
        )


        st.bar_chart(
            probability_chart_df
        )


        # ====================================================
        # DECISION CONTEXT
        # ====================================================

        st.subheader(
            "Decision Context"
        )


        if confidence_gate_triggered:

            st.write(
                "• The original ML recommendation was below "
                "the minimum confidence threshold, so the "
                "controller preserved the appliance's current "
                "operating state before safety validation."
            )


        if safety_override:

            st.write(
                "• The confidence-gated action violated a "
                "deterministic operating constraint and was "
                "therefore corrected by the safety layer."
            )


        decision_reasons = (
            create_decision_context(

                input_df.iloc[
                    0
                ],

                final_action
            )
        )


        for reason in decision_reasons:

            st.write(
                f"• {reason}"
            )


        st.caption(
            "These statements summarize the current decision "
            "context. Formal model explainability was evaluated "
            "separately using XGBoost feature importance and "
            "SHAP analysis."
        )


        # ====================================================
        # GLOBAL EXPLAINABILITY
        # ====================================================

        if (
            importance_df is not None
            and len(
                importance_df
            ) > 0
        ):

            st.subheader(
                "Model Explainability — Top Global Drivers"
            )


            top_features = (
                importance_df
                .head(
                    10
                )
                .copy()
            )


            if "Feature_Group" in top_features.columns:

                top_features[
                    "Feature_Group"
                ] = (
                    top_features[
                        "Feature_Group"
                    ]
                    .astype(
                        str
                    )
                )


            if "Importance" in top_features.columns:

                top_features[
                    "Importance"
                ] = pd.to_numeric(
                    top_features[
                        "Importance"
                    ],
                    errors="coerce"
                ).fillna(
                    0.0
                )


            st.dataframe(

                top_features,

                width="stretch",

                hide_index=True
            )


            if (
                "Feature_Group"
                in top_features.columns
                and "Importance"
                in top_features.columns
            ):

                feature_chart_df = (
                    top_features[
                        [
                            "Feature_Group",
                            "Importance"
                        ]
                    ]
                    .set_index(
                        "Feature_Group"
                    )
                )


                st.bar_chart(
                    feature_chart_df
                )


        # ====================================================
        # MQTT COMMAND PREVIEW
        # ====================================================

        st.subheader(
            "3. MQTT Command Preview"
        )


        mqtt_appliance_name = (
            appliance
            .lower()
            .replace(
                " ",
                "_"
            )
        )


        mqtt_topic = (
            f"smart_home/component3/"
            f"{mqtt_appliance_name}/command"
        )


        mqtt_payload = (
            create_mqtt_payload(

                appliance,
                ml_action,
                gated_action,
                final_action,
                confidence,
                confidence_gate_triggered,
                confidence_gate_reason,
                safety_override,
                safety_rule
            )
        )


        mqtt_status_col1, mqtt_status_col2 = (
            st.columns(
                2
            )
        )


        with mqtt_status_col1:

            if confidence_gate_triggered:

                st.metric(
                    "Confidence Gate",
                    "TRIGGERED"
                )

            else:

                st.metric(
                    "Confidence Gate",
                    "PASSED"
                )


        with mqtt_status_col2:

            if safety_override:

                st.metric(
                    "Safety Layer",
                    "OVERRIDE"
                )

            else:

                st.metric(
                    "Safety Layer",
                    "PASSED"
                )


        mqtt_col1, mqtt_col2 = (
            st.columns(
                2
            )
        )


        with mqtt_col1:

            st.write(
                "**MQTT Topic**"
            )

            st.code(
                mqtt_topic,
                language="text"
            )


        with mqtt_col2:

            st.write(
                "**Hardware Output Mapping**"
            )

            st.code(
                RELAY_MAPPING[
                    appliance
                ],
                language="text"
            )


        st.write(
            "**MQTT JSON Payload**"
        )


        st.code(
            json.dumps(
                mqtt_payload,
                indent=4
            ),
            language="json"
        )


        # ====================================================
        # COMPLETE DECISION INPUT
        # ====================================================

        with st.expander(
            "View Complete Decision Input"
        ):

            display_columns = [

                "appliance_category",
                "appliance_active",
                "appliance_mean_w",

                "user_profile",
                "user_flexibility",

                "occupancy",

                "tariff_band",
                "tariff_cost_index",

                "predicted_load_w",
                "predicted_load_level",

                "normalized_predicted_load",
                "normalized_appliance_power",

                "off_duration_minutes",
                "state_duration_minutes",
                "on_duration_minutes",

                "max_off_minutes",
                "max_delay_minutes",

                "priority_level",
                "interruptible",
                "critical_flag",
                "flexible_flag",
                "high_power_short_use",

                "time_period",
                "hour"
            ]


            available_display_columns = [

                column
                for column in display_columns
                if column in input_df.columns
            ]


            summary_records = []


            for column in available_display_columns:

                value = input_df.loc[
                    0,
                    column
                ]


                if pd.isna(
                    value
                ):

                    display_value = ""

                elif isinstance(
                    value,
                    (
                        float,
                        np.floating
                    )
                ):

                    display_value = (
                        f"{float(value):.4f}"
                    )

                else:

                    display_value = str(
                        value
                    )


                summary_records.append({

                    "Input":
                        str(
                            column
                        ),

                    "Value":
                        display_value
                })


            input_summary_df = pd.DataFrame(
                summary_records,
                columns=[
                    "Input",
                    "Value"
                ]
            )


            input_summary_df[
                "Input"
            ] = (
                input_summary_df[
                    "Input"
                ]
                .astype(
                    str
                )
            )


            input_summary_df[
                "Value"
            ] = (
                input_summary_df[
                    "Value"
                ]
                .astype(
                    str
                )
            )


            st.dataframe(

                input_summary_df,

                width="stretch",

                hide_index=True
            )


    except Exception as error:

        st.error(
            "The scheduling decision could not be generated."
        )

        st.exception(
            error
        )


# ============================================================
# RESEARCH VALIDATION SUMMARY
# ============================================================

st.divider()

st.header(
    "Research Validation Summary"
)


(
    validation_col1,
    validation_col2,
    validation_col3,
    validation_col4
) = st.columns(
    4
)


with validation_col1:

    st.metric(
        "XGBoost Accuracy",
        "84.7%"
    )


with validation_col2:

    st.metric(
        "Macro F1",
        "84.3%"
    )


with validation_col3:

    st.metric(
        "Peak Exposure Reduction",
        "17.5%"
    )


with validation_col4:

    st.metric(
        "Safety Compliance",
        "100%"
    )


st.caption(
    "Operational energy and normalized tariff-cost results "
    "are simulation-based PP1 estimates and should not be "
    "interpreted as measured household electricity-bill savings."
)