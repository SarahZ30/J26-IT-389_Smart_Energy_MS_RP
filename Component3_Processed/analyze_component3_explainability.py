import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

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

OUTPUT_DIR = "explainability_output"

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

SHAP_SAMPLE_SIZE = 3000

LOCAL_EXPLANATION_RECORDS = 30


# ============================================================
# 1. START
# ============================================================

print("============================================")
print("COMPONENT 3 - MODEL EXPLAINABILITY")
print("============================================")


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading V2 dataset...")

df = pd.read_csv(
    DATA_FILE
)

print(
    f"Total records: {len(df):,}"
)

print(
    f"Unique houses: {df['house_id'].nunique()}"
)


# ============================================================
# 3. LOAD MODEL
# ============================================================

print("\nLoading selected model...")

pipeline = joblib.load(
    MODEL_FILE
)

label_encoder = joblib.load(
    LABEL_ENCODER_FILE
)

preprocessor = pipeline.named_steps[
    "preprocessor"
]

classifier = pipeline.named_steps[
    "classifier"
]

print(
    "Selected model loaded successfully."
)


# ============================================================
# 4. RECREATE TEST SPLIT
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

print(
    f"\nTest records: {len(test_df):,}"
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
# 5. MODEL FEATURE INPUTS
# ============================================================

# Extract exact column lists from trained preprocessor

categorical_features = []

numeric_features = []

for name, transformer, columns in preprocessor.transformers_:

    if name == "categorical":

        categorical_features = list(
            columns
        )

    elif name == "numeric":

        numeric_features = list(
            columns
        )


raw_feature_columns = (
    categorical_features
    + numeric_features
)

print(
    f"\nRaw model features: "
    f"{len(raw_feature_columns)}"
)


# ============================================================
# 6. TRANSFORM TEST DATA
# ============================================================

print(
    "\nTransforming test data..."
)

X_test_raw = test_df[
    raw_feature_columns
].copy()

X_test_transformed = (
    preprocessor.transform(
        X_test_raw
    )
)


# Convert sparse matrix if necessary
if hasattr(
    X_test_transformed,
    "toarray"
):

    # Do NOT convert entire 44k matrix if large.
    pass


# ============================================================
# 7. EXTRACT TRANSFORMED FEATURE NAMES
# ============================================================

transformed_feature_names = []

for name, transformer, columns in preprocessor.transformers_:

    if name == "categorical":

        encoded_names = (
            transformer
            .get_feature_names_out(
                columns
            )
        )

        transformed_feature_names.extend(
            encoded_names.tolist()
        )

    elif name == "numeric":

        transformed_feature_names.extend(
            list(
                columns
            )
        )


print(
    f"Transformed feature count: "
    f"{len(transformed_feature_names)}"
)


# ============================================================
# 8. GLOBAL XGBOOST FEATURE IMPORTANCE
# ============================================================

print(
    "\nCalculating global feature importance..."
)

importance_values = (
    classifier.feature_importances_
)

importance_df = pd.DataFrame({

    "Feature":
        transformed_feature_names,

    "Importance":
        importance_values

})

importance_df = (
    importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


importance_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "global_feature_importance.csv"
    ),
    index=False
)


print(
    "\nTop 20 transformed features:"
)

print(
    importance_df
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# 9. GLOBAL IMPORTANCE PLOT
# ============================================================

top20 = (
    importance_df
    .head(20)
    .sort_values(
        "Importance",
        ascending=True
    )
)

fig, ax = plt.subplots(
    figsize=(10, 8)
)

ax.barh(
    top20[
        "Feature"
    ],
    top20[
        "Importance"
    ]
)

ax.set_xlabel(
    "XGBoost Feature Importance"
)

ax.set_title(
    "Top 20 Features Driving Appliance Scheduling"
)

fig.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "global_feature_importance.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 10. AGGREGATE ONE-HOT FEATURES BACK TO RAW VARIABLES
# ============================================================

print(
    "\nAggregating importance to raw feature groups..."
)

raw_importance = {}


for feature_name, importance in zip(
    transformed_feature_names,
    importance_values
):

    matched = False


    # --------------------------------------------------------
    # Match one-hot categorical features
    # --------------------------------------------------------

    for raw_feature in categorical_features:

        prefix = (
            raw_feature
            + "_"
        )

        if feature_name.startswith(
            prefix
        ):

            raw_importance[
                raw_feature
            ] = (
                raw_importance.get(
                    raw_feature,
                    0
                )
                + float(
                    importance
                )
            )

            matched = True
            break


    # --------------------------------------------------------
    # Numeric feature
    # --------------------------------------------------------

    if not matched:

        raw_importance[
            feature_name
        ] = (
            raw_importance.get(
                feature_name,
                0
            )
            + float(
                importance
            )
        )


raw_importance_df = pd.DataFrame(

    list(
        raw_importance.items()
    ),

    columns=[
        "Feature_Group",
        "Importance"
    ]

)

raw_importance_df = (
    raw_importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


raw_importance_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "raw_feature_group_importance.csv"
    ),
    index=False
)


print(
    "\nTop raw feature groups:"
)

print(
    raw_importance_df
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# 11. RAW FEATURE GROUP PLOT
# ============================================================

top_raw = (
    raw_importance_df
    .head(20)
    .sort_values(
        "Importance",
        ascending=True
    )
)

fig, ax = plt.subplots(
    figsize=(10, 8)
)

ax.barh(
    top_raw[
        "Feature_Group"
    ],
    top_raw[
        "Importance"
    ]
)

ax.set_xlabel(
    "Aggregated Feature Importance"
)

ax.set_title(
    "Most Important Scheduling Input Variables"
)

fig.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "raw_feature_group_importance.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 12. PREPARE SHAP SAMPLE
# ============================================================

print(
    "\nPreparing SHAP sample..."
)

sample_size = min(
    SHAP_SAMPLE_SIZE,
    len(test_df)
)

shap_sample_df = test_df.sample(
    n=sample_size,
    random_state=RANDOM_STATE
).copy()


X_shap_raw = shap_sample_df[
    raw_feature_columns
].copy()


X_shap_transformed = (
    preprocessor.transform(
        X_shap_raw
    )
)


if hasattr(
    X_shap_transformed,
    "toarray"
):

    X_shap_dense = (
        X_shap_transformed
        .toarray()
    )

else:

    X_shap_dense = np.asarray(
        X_shap_transformed
    )


print(
    f"SHAP sample records: "
    f"{len(X_shap_dense):,}"
)


# ============================================================
# 13. CREATE SHAP EXPLAINER
# ============================================================

print(
    "\nRunning SHAP explainability..."
)

explainer = shap.TreeExplainer(
    classifier
)

shap_values = explainer.shap_values(
    X_shap_dense
)


# ============================================================
# 14. NORMALIZE SHAP OUTPUT FORMAT
# ============================================================

# SHAP versions may return:
#
# 1. list[class] of arrays
# 2. ndarray with shape:
#    samples x features x classes
# 3. Explanation.values with similar shape


if hasattr(
    shap_values,
    "values"
):

    shap_array = (
        shap_values.values
    )

else:

    shap_array = shap_values


if isinstance(
    shap_array,
    list
):

    # classes x samples x features
    shap_class_arrays = (
        shap_array
    )

else:

    shap_array = np.asarray(
        shap_array
    )

    if shap_array.ndim == 3:

        # Detect samples x features x classes
        if (
            shap_array.shape[0]
            == len(X_shap_dense)
        ):

            shap_class_arrays = [

                shap_array[
                    :,
                    :,
                    class_index
                ]

                for class_index
                in range(
                    shap_array.shape[2]
                )

            ]

        else:

            # classes x samples x features
            shap_class_arrays = [

                shap_array[
                    class_index
                ]

                for class_index
                in range(
                    shap_array.shape[0]
                )

            ]

    elif shap_array.ndim == 2:

        shap_class_arrays = [
            shap_array
        ]

    else:

        raise ValueError(
            "Unsupported SHAP output shape: "
            f"{shap_array.shape}"
        )


# ============================================================
# 15. GLOBAL MEAN ABSOLUTE SHAP IMPORTANCE
# ============================================================

class_mean_abs = []

for class_array in shap_class_arrays:

    class_mean_abs.append(
        np.abs(
            class_array
        )
        .mean(
            axis=0
        )
    )


global_mean_abs_shap = (
    np.mean(
        np.vstack(
            class_mean_abs
        ),
        axis=0
    )
)


shap_importance_df = pd.DataFrame({

    "Feature":
        transformed_feature_names,

    "Mean_Absolute_SHAP":
        global_mean_abs_shap

})

shap_importance_df = (
    shap_importance_df
    .sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


shap_importance_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "global_shap_importance.csv"
    ),
    index=False
)


# ============================================================
# 16. GLOBAL SHAP IMPORTANCE PLOT
# ============================================================

top_shap = (
    shap_importance_df
    .head(20)
    .sort_values(
        "Mean_Absolute_SHAP",
        ascending=True
    )
)

fig, ax = plt.subplots(
    figsize=(10, 8)
)

ax.barh(
    top_shap[
        "Feature"
    ],
    top_shap[
        "Mean_Absolute_SHAP"
    ]
)

ax.set_xlabel(
    "Mean Absolute SHAP Value"
)

ax.set_title(
    "Global SHAP Importance - Selected XGBoost Model"
)

fig.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "global_shap_importance.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 17. CLASS-SPECIFIC SHAP IMPORTANCE
# ============================================================

print(
    "\nSaving class-specific SHAP results..."
)

for class_index, class_array in enumerate(
    shap_class_arrays
):

    if class_index >= len(
        label_encoder.classes_
    ):

        continue


    class_name = (
        label_encoder.classes_[
            class_index
        ]
    )


    mean_abs_class = (
        np.abs(
            class_array
        )
        .mean(
            axis=0
        )
    )


    class_df = pd.DataFrame({

        "Feature":
            transformed_feature_names,

        "Mean_Absolute_SHAP":
            mean_abs_class

    })


    class_df = (
        class_df
        .sort_values(
            "Mean_Absolute_SHAP",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    class_df.to_csv(
        os.path.join(
            RESULT_DIR,
            f"shap_{class_name}_importance.csv"
        ),
        index=False
    )


    top_class = (
        class_df
        .head(15)
        .sort_values(
            "Mean_Absolute_SHAP",
            ascending=True
        )
    )


    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    ax.barh(
        top_class[
            "Feature"
        ],
        top_class[
            "Mean_Absolute_SHAP"
        ]
    )

    ax.set_xlabel(
        "Mean Absolute SHAP Value"
    )

    ax.set_title(
        f"Features Influencing {class_name}"
    )

    fig.tight_layout()


    plt.savefig(
        os.path.join(
            PLOT_DIR,
            f"shap_{class_name}_importance.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 18. PREDICTIONS FOR LOCAL EXPLANATIONS
# ============================================================

sample_predictions = pipeline.predict(
    X_shap_raw
)

sample_probabilities = pipeline.predict_proba(
    X_shap_raw
)

predicted_labels = (
    label_encoder.inverse_transform(
        sample_predictions.astype(int)
    )
)


# ============================================================
# 19. SELECT LOCAL EXPLANATION RECORDS
# ============================================================

local_n = min(
    LOCAL_EXPLANATION_RECORDS,
    len(shap_sample_df)
)

local_indices = np.linspace(
    0,
    len(shap_sample_df) - 1,
    num=local_n,
    dtype=int
)


local_records = []


for sample_position in local_indices:

    original_row = (
        shap_sample_df
        .iloc[
            sample_position
        ]
    )

    predicted_class_id = int(
        sample_predictions[
            sample_position
        ]
    )

    predicted_action = (
        predicted_labels[
            sample_position
        ]
    )

    confidence = float(
        sample_probabilities[
            sample_position,
            predicted_class_id
        ]
    )


    if predicted_class_id >= len(
        shap_class_arrays
    ):

        continue


    local_shap = (
        shap_class_arrays[
            predicted_class_id
        ][
            sample_position
        ]
    )


    top_indices = (
        np.argsort(
            np.abs(
                local_shap
            )
        )[
            -5:
        ][
            ::-1
        ]
    )


    top_features = [
        transformed_feature_names[
            i
        ]
        for i in top_indices
    ]

    top_values = [
        float(
            local_shap[
                i
            ]
        )
        for i in top_indices
    ]


    local_records.append({

        "house_id":
            original_row[
                "house_id"
            ],

        "Time":
            original_row.get(
                "Time",
                ""
            ),

        "appliance_category":
            original_row[
                "appliance_category"
            ],

        "user_profile":
            original_row[
                "user_profile"
            ],

        "tariff_band":
            original_row[
                "tariff_band"
            ],

        "predicted_load_level":
            original_row[
                "predicted_load_level"
            ],

        "user_flexibility":
            original_row[
                "user_flexibility"
            ],

        "occupancy":
            original_row[
                "occupancy"
            ],

        "actual_target":
            original_row[
                "target_action"
            ],

        "predicted_action":
            predicted_action,

        "prediction_confidence":
            confidence,

        "top_feature_1":
            top_features[0],

        "top_feature_1_shap":
            top_values[0],

        "top_feature_2":
            top_features[1],

        "top_feature_2_shap":
            top_values[1],

        "top_feature_3":
            top_features[2],

        "top_feature_3_shap":
            top_values[2],

        "top_feature_4":
            top_features[3],

        "top_feature_4_shap":
            top_values[3],

        "top_feature_5":
            top_features[4],

        "top_feature_5_shap":
            top_values[4]

    })


local_explanations_df = pd.DataFrame(
    local_records
)


local_explanations_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "local_prediction_explanations.csv"
    ),
    index=False
)


# ============================================================
# 20. CONFIDENCE ANALYSIS
# ============================================================

print(
    "\nCalculating prediction confidence..."
)

all_probabilities = pipeline.predict_proba(
    X_test_raw
)

all_predictions = pipeline.predict(
    X_test_raw
)

max_probabilities = (
    all_probabilities.max(
        axis=1
    )
)


confidence_df = pd.DataFrame({

    "Prediction_Confidence":
        max_probabilities,

    "Correct":
        (
            label_encoder.inverse_transform(
                all_predictions.astype(int)
            )
            ==
            test_df[
                "target_action"
            ].values
        ).astype(int),

    "Decision_Margin":
        test_df[
            "decision_margin"
        ].values

})


confidence_df[
    "Confidence_Band"
] = pd.cut(

    confidence_df[
        "Prediction_Confidence"
    ],

    bins=[
        0,
        0.60,
        0.75,
        0.90,
        1.00
    ],

    labels=[
        "Low",
        "Medium",
        "High",
        "Very_High"
    ],

    include_lowest=True
)


confidence_summary = (
    confidence_df.groupby(
        "Confidence_Band",
        observed=False
    )
    .agg(

        Records=(
            "Correct",
            "size"
        ),

        Accuracy=(
            "Correct",
            "mean"
        ),

        Average_Confidence=(
            "Prediction_Confidence",
            "mean"
        ),

        Average_Decision_Margin=(
            "Decision_Margin",
            "mean"
        )

    )
    .reset_index()
)


confidence_summary.to_csv(
    os.path.join(
        RESULT_DIR,
        "prediction_confidence_summary.csv"
    ),
    index=False
)


# ============================================================
# 21. CONFIDENCE PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 6)
)

ax.bar(
    confidence_summary[
        "Confidence_Band"
    ].astype(str),
    confidence_summary[
        "Accuracy"
    ]
)

ax.set_ylim(
    0,
    1.05
)

ax.set_ylabel(
    "Prediction Accuracy"
)

ax.set_xlabel(
    "Prediction Confidence Band"
)

ax.set_title(
    "Prediction Accuracy by Confidence Level"
)

fig.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "prediction_confidence_accuracy.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 22. SUMMARY
# ============================================================

top_global_feature = (
    raw_importance_df.iloc[0][
        "Feature_Group"
    ]
)

top_global_importance = (
    raw_importance_df.iloc[0][
        "Importance"
    ]
)

top_shap_feature = (
    shap_importance_df.iloc[0][
        "Feature"
    ]
)

top_shap_value = (
    shap_importance_df.iloc[0][
        "Mean_Absolute_SHAP"
    ]
)


summary_df = pd.DataFrame([{

    "Model":
        "XGBoost_Personalized_Scheduler",

    "Test_Records":
        len(test_df),

    "Raw_Feature_Count":
        len(raw_feature_columns),

    "Transformed_Feature_Count":
        len(
            transformed_feature_names
        ),

    "SHAP_Sample_Size":
        sample_size,

    "Top_Aggregated_Feature":
        top_global_feature,

    "Top_Aggregated_Importance":
        top_global_importance,

    "Top_SHAP_Feature":
        top_shap_feature,

    "Top_SHAP_Importance":
        top_shap_value

}])


summary_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "explainability_summary.csv"
    ),
    index=False
)


# ============================================================
# 23. FINAL TERMINAL OUTPUT
# ============================================================

print("\n")
print("============================================")
print("EXPLAINABILITY ANALYSIS COMPLETED")
print("============================================")


print(
    f"\nTop aggregated feature: "
    f"{top_global_feature}"
)

print(
    f"Top aggregated importance: "
    f"{top_global_importance:.6f}"
)

print(
    f"\nTop SHAP feature: "
    f"{top_shap_feature}"
)

print(
    f"Mean absolute SHAP: "
    f"{top_shap_value:.6f}"
)


print(
    "\nTop 10 aggregated features:"
)

print(
    raw_importance_df
    .head(10)
    .to_string(
        index=False
    )
)


print(
    "\nPrediction-confidence analysis:"
)

print(
    confidence_summary.to_string(
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
    "\nExplainability analysis completed."
)