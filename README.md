# Component 3 – Personalized Appliance Scheduling for Smart Home Energy Management

## Project Information

**Research Project:** AI-Based Smart Energy Management System using IoT, Edge Computing and Machine Learning  
**Project ID:** J26-IT-389  
**Component:** Component 3  
**Student:** Fernando H.G.  
**Student ID:** IT23310582

---

# 1. Overview

Component 3 is the **Personalized Appliance Scheduling and Optimization** component of the overall smart energy management system.

The overall research system can be understood using the following pipeline:

```text
Sense
  ↓
Clean / Validate Data
  ↓
Predict Future Energy Demand
  ↓
Decide What Appliances Should Do
  ↓
Control Appliances
  ↓
Monitor Performance
  ↓
Adapt / Learn
```

The four main research components have different responsibilities:

| Component   | Main Question                       | Main Responsibility                                |
| ----------- | ----------------------------------- | -------------------------------------------------- |
| Component 1 | Can we trust the data?              | Edge data processing / anomaly detection           |
| Component 2 | What will happen next?              | Energy / load forecasting                          |
| Component 3 | What should the appliance do?       | Personalized appliance scheduling and optimization |
| Component 4 | Is the AI still accurate over time? | Adaptive / federated intelligence                  |

Component 3 is therefore the **decision-making layer**.

It receives information such as:

- appliance category
- appliance ON/OFF state
- current appliance power
- household load
- predicted future load
- user preference
- user flexibility
- tariff period
- occupancy
- environmental context
- appliance operating constraints

and predicts one of six possible scheduling actions:

```text
TURN_ON
TURN_OFF
DELAY
ECO_MODE
PRIORITY
LOAD_BALANCE
```

The final PP1 runtime flow is:

```text
Input Context
   ↓
XGBoost Recommendation
   ↓
Prediction Confidence
   ↓
60% Confidence Gate
   ↓
Rule-Based Safety Layer
   ↓
Final Safe Action
   ↓
MQTT-Ready Command
```

---

# 2. Main Codebase Structure

The Component 3 codebase is organized approximately as follows:

```text
Component_3/
│
├── appliances+energy+prediction/
│
├── Component3_REFIT/
│   ├── audit_refit.py
│   ├── build_appliance_summary.py
│   ├── CLEAN_House1.csv
│   ├── CLEAN_House2.csv
│   ├── ...
│   ├── CLEAN_House21.csv
│   ├── CLEAN_READ_ME_081116.txt
│   ├── MetaData_Tables.xlsx
│   ├── refit_appliance_category_summary.csv
│   ├── refit_appliance_overall_summary.csv
│   └── refit_house_summary.csv
│
└── Component3_Processed/
    │
    ├── ablation_final_output/
    ├── ablation_output/
    ├── explainability_output/
    ├── model_output/
    ├── operational_benefit_output/
    ├── output/
    ├── safety_layer_output/
    ├── v2_model_output/
    ├── v2_output/
    │
    ├── analyze_component3_explainability.py
    ├── build_component3_dataset.py
    ├── build_component3_v2.py
    ├── component3_dashboard.py
    ├── energydata_complete.csv
    ├── evaluate_component3_operational_benefits.py
    ├── evaluate_component3_safety_layer.py
    ├── generate_scheduling_labels.py
    ├── prepare_component3_base_dataset.py
    ├── prepare_feature_fusion.py
    ├── requirements.txt
    ├── run_component3_ablation_study.py
    ├── run_component3_ablation_study_final.py
    ├── train_component3_models.py
    └── train_component3_v2_models.py
```

---

# 3. Purpose of the Main Folders

## `Component3_REFIT/`

This folder contains the main REFIT source data and the initial dataset audit work.

Purpose:

- inspect REFIT data structure
- verify household files
- inspect metadata
- map appliance channels
- summarize appliance availability
- confirm useful households and categories

Important scripts:

```text
audit_refit.py
build_appliance_summary.py
```

---

## `appliances+energy+prediction/`

This folder contains the second dataset used for environmental context.

The main purpose of this dataset is to provide:

- indoor temperature
- indoor humidity
- outdoor environmental information

It is not used as the main appliance-level source.

---

## `Component3_Processed/`

This is the main research development workspace.

All major Component 3 research stages are implemented here:

```text
REFIT preprocessing
↓
Base dataset
↓
Feature fusion
↓
V1 scheduling labels
↓
V1 model training
↓
V2 advanced dataset
↓
V2 model training
↓
Ablation study
↓
Safety evaluation
↓
Operational benefit evaluation
↓
Explainability
↓
Confidence analysis
↓
Streamlit dashboard
```

---

# 4. Why REFIT Was Selected

Component 3 requires **appliance-level electrical behavior**.

A household-level energy dataset alone is not sufficient because the decision system must distinguish between different appliances.

For example:

```text
Refrigerator
Washing Machine
Dishwasher
Kettle
```

REFIT was selected because it contains:

- aggregate household electricity
- multiple appliance channels
- real household measurements
- long-term appliance behavior
- approximately 8-second active-power readings
- data from multiple households

REFIT contains data from 20 households.

House 14 is intentionally absent in the original published dataset.

---

# 5. REFIT Dataset Audit

Before model development, the REFIT data was audited.

Relevant location:

```text
Component3_REFIT/
```

Important files:

```text
CLEAN_READ_ME_081116.txt
MetaData_Tables.xlsx
audit_refit.py
build_appliance_summary.py
```

The audit stage checked:

- available household files
- file sizes
- appliance channels
- metadata mappings
- appliance names
- issue flags
- household coverage
- appliance category availability

The generated summary files include:

```text
refit_appliance_category_summary.csv
refit_appliance_overall_summary.csv
refit_house_summary.csv
```

---

# 6. House 1 Data Issue

During REFIT processing, a problem was identified with the working copy of House 1.

The file had been incorrectly truncated during data handling.

House 1 contains millions of records, so normal spreadsheet handling is not suitable.

The issue was handled by:

```text
Detect incorrect House 1 working copy
        ↓
Re-extract original REFIT House 1
        ↓
Verify full file
        ↓
Continue preprocessing
```

The correct House 1 contained approximately:

```text
6.96 million rows
```

with approximately:

```text
6.90 million valid rows
```

after filtering invalid issue records.

This step was important because incomplete House 1 data could affect downstream preprocessing and model evaluation.

---

# 7. REFIT Cleaning and Preprocessing

REFIT already provides cleaned data, but Component 3 still required research-specific preprocessing.

The main preprocessing flow was:

```text
CLEAN_House*.csv
        ↓
Validate household files
        ↓
Use metadata to map Appliance1...Appliance9
        ↓
Remove / exclude Issues = 1
        ↓
Normalize appliance names
        ↓
Select research appliance categories
        ↓
Resample to 15-minute intervals
        ↓
Generate appliance/load features
```

Relevant scripts:

```text
Component3_REFIT/audit_refit.py
Component3_REFIT/build_appliance_summary.py
Component3_Processed/prepare_component3_base_dataset.py
```

---

# 8. Why the ~8-Second REFIT Data Was Converted to 15 Minutes

REFIT contains high-frequency active-power readings at approximately 8-second intervals.

Using the raw high-frequency data directly would:

- create very large datasets
- increase processing time
- provide unnecessary detail for scheduling
- make long-term model development harder
- complicate alignment with future forecasting intervals

Therefore the data was converted into **15-minute scheduling intervals**.

Main script:

```text
Component3_Processed/prepare_component3_base_dataset.py
```

Useful VS Code search terms:

```text
resample
15min
```

Conceptually:

```text
~8-second readings
      ↓
time-based aggregation
      ↓
15-minute intervals
      ↓
mean / max / active-state / load features
```

The 15-minute interval is also useful for temporal engineering because:

```text
4 × 15 minutes = 1 hour
```

Therefore a 1-hour rolling window can be created using four records.

---

# 9. Selected Appliance Categories

Four appliance categories were selected:

| Appliance       | Main Scheduling Behavior           |
| --------------- | ---------------------------------- |
| Refrigeration   | Critical / high-priority appliance |
| Washing Machine | Flexible but cycle-based           |
| Dishwasher      | Delayable / tariff-aware           |
| Kettle          | Short-duration high-power          |

These four appliances were selected because they represent different scheduling behaviors.

The system is not permanently limited to four appliances.

More categories can be added later.

Approximate processed REFIT interval counts:

```text
Refrigeration     1,483,279
Washing Machine     880,116
Dishwasher          688,733
Kettle              682,797
```

The REFIT-based Component 3 base dataset contained roughly:

```text
3.73 million records
```

before later controlled sampling.

---

# 10. Base Dataset

The main base dataset was prepared using:

```text
Component3_Processed/prepare_component3_base_dataset.py
```

This stage generated appliance and household scheduling features such as:

```text
appliance_category
appliance_mean_w
appliance_max_w
appliance_active
household aggregate load
appliance_load_ratio
critical_flag
flexible_flag
high_power_short_use
hour
day_of_week
weekend
time_period
household_load_level
```

The purpose of this stage was to convert REFIT into a scheduling-friendly dataset.

---

# 11. REFIT Limitations

REFIT was a strong source for appliance-level electrical behavior.

However, REFIT did not provide all information required for Component 3.

Missing variables included:

```text
temperature
humidity
occupancy
tariff
user preference
user flexibility
future predicted load
scheduling target action
```

Therefore REFIT alone was not a perfect dataset for the research scenario.

---

# 12. Second Dataset – Appliances Energy Prediction

To obtain realistic environmental context, a second dataset was used.

Working file:

```text
Component3_Processed/energydata_complete.csv
```

The dataset contains approximately:

```text
19,735 rows
29 columns
```

Environmental variables include:

```text
T1 ... T9
RH1 ... RH9
outside temperature
outside humidity
environmental context variables
```

The dataset could not replace REFIT because its `Appliances` column represents combined appliance energy.

It does not provide appliance identity such as:

```text
Refrigerator
Washing Machine
Dishwasher
Kettle
```

Therefore:

```text
REFIT
→ real appliance electrical behavior

Appliances Energy Prediction
→ realistic environmental context
```

---

# 13. How the Two Datasets Were Combined

The two datasets were **not directly joined by timestamp**.

They represent different households and different recording periods.

A direct timestamp merge would incorrectly imply that:

```text
REFIT household reading
=
same household/time as environmental reading
```

which would be scientifically incorrect.

Instead, environmental data was used as **contextual augmentation**.

Main script:

```text
Component3_Processed/prepare_feature_fusion.py
```

Conceptually:

```text
REFIT electrical data
        +
environmental context
        ↓
Component 3 fused feature table
```

This should be described as:

> Environmental contextual augmentation, not synchronized same-house measurement fusion.

---

# 14. Missing Variables Added for PP1

Even after using REFIT and the environmental dataset, several Component 3 variables were still unavailable.

For PP1, these variables were created transparently.

| Variable         | PP1 Source               | Future Source                   |
| ---------------- | ------------------------ | ------------------------------- |
| Occupancy        | Controlled scenario      | PIR / live occupancy sensor     |
| User Preference  | User-profile scenario    | Real user preference            |
| User Flexibility | Controlled profile value | Learned/user-configured         |
| Tariff           | Off-Peak / Day / Peak    | Final tariff source             |
| Predicted Load   | History-based proxy      | Component 2                     |
| Target Action    | Utility/policy-derived   | Future deployed/user validation |

---

# 15. Occupancy Scenario

Occupancy was not directly available from REFIT.

For PP1, occupancy was generated using controlled time-based behavior.

Future integration should replace this with:

```text
PIR sensor
or
Component 1 occupancy input
```

---

# 16. Tariff Scenario

Temporary PP1 tariff bands were created:

```text
00:00 – 06:00 → Off_Peak
18:00 – 22:00 → Peak
Other periods  → Day
```

Normalized experimental cost indices:

```text
Off_Peak = 0.30
Day      = 0.60
Peak     = 1.00
```

These are **experimental normalized indices**.

They are not official Sri Lankan electricity tariffs.

---

# 17. Version 1 Dataset

The first version of the scheduling dataset used deterministic IF/ELSE-style policy rules.

Main script:

```text
Component3_Processed/generate_scheduling_labels.py
```

Model training:

```text
Component3_Processed/train_component3_models.py
```

Example concept:

```text
if refrigerator is critical:
    PRIORITY

if appliance is flexible and tariff is peak:
    DELAY

if household load is high:
    LOAD_BALANCE
```

Target classes:

```text
TURN_ON
TURN_OFF
DELAY
ECO_MODE
PRIORITY
LOAD_BALANCE
```

Approximate class distribution:

```text
TURN_OFF       100,000
LOAD_BALANCE    28,105
PRIORITY        25,000
ECO_MODE        24,112
TURN_ON         15,996
DELAY            6,787
```

---

# 18. V1 Model Results

The following models were trained:

```text
Decision Tree
Random Forest
XGBoost
```

The results were approximately:

```text
Decision Tree ≈ 99.99%
Random Forest = 100%
XGBoost       = 100%
```

At first, these results looked excellent.

However, scientifically this was a problem.

---

# 19. Why V1 Was Not Used as the Final Dataset

The target labels were generated using deterministic rules.

Many of the same variables used in the rules were also given as model inputs.

Therefore the model could simply learn the rule logic.

In other words:

```text
IF/ELSE target logic
        ↓
same variables given to ML
        ↓
very easy classification
        ↓
near-100% accuracy
```

This was not considered a strong final research design.

The V1 experiment was retained as:

```text
Baseline policy-learning validation
```

The problem was then redesigned into V2.

---

# 20. Version 2 Dataset

The advanced V2 dataset was created using:

```text
Component3_Processed/build_component3_v2.py
```

Main output:

```text
Component3_Processed/v2_output/component3_v2_dataset.csv
```

V2 contains:

```text
250,000 records
20 households
6 scheduling actions
```

Target distribution:

| Action       | Records | Percentage |
| ------------ | ------: | ---------: |
| TURN_OFF     |  66,678 |     26.67% |
| TURN_ON      |  56,377 |     22.55% |
| ECO_MODE     |  47,701 |     19.08% |
| LOAD_BALANCE |  40,143 |     16.06% |
| DELAY        |  29,582 |     11.83% |
| PRIORITY     |   9,519 |      3.81% |

---

# 21. V2 Main Improvements

V2 introduced:

```text
Temporal history
Load lags
Rolling statistics
Predicted-load proxy
User profiles
User flexibility
Tariff context
Occupancy
Appliance constraints
Multi-objective utility scoring
Controlled ambiguity
Dataset provenance
```

The main conceptual difference is:

```text
V1:
IF condition
→ direct label

V2:
multiple competing objectives
→ utility scores
→ best action
```

---

# 22. V2 Temporal Features

The following history features were created:

```text
load_lag_1
load_lag_2
load_lag_4
previous_appliance_active
rolling_load_mean_1h
rolling_load_max_1h
state_duration_minutes
on_duration_minutes
off_duration_minutes
```

These were generated chronologically per:

```text
household
+
appliance category
```

The purpose was to help the model understand recent behavior.

For example:

```text
Current load high
+
past hour also high
→ sustained high demand
```

versus:

```text
Current load high
+
previous load low
→ temporary spike
```

---

# 23. V2 Predicted Load Proxy

Component 3 ideally receives future load information from Component 2.

At PP1, Component 2 integration was not ready.

Therefore a temporary history-based forecast proxy was created.

Location:

```text
Component3_Processed/build_component3_v2.py
```

Useful search terms:

```text
load_lag_1
load_lag_2
load_lag_4
rolling_load_mean_1h
predicted_load_w
```

Conceptually:

```text
Recent load history
     +
Lag values
     +
Rolling average
     +
Small controlled uncertainty
     ↓
predicted_load_w
```

This was then converted into:

```text
Low
Medium
High
```

predicted-load levels.

This is only a PP1 proxy.

Later:

```text
Component 2 forecast
       ↓
Component 3 predicted_load_w
```

should replace it.

---

# 24. V2 Personalization

Four user profiles were created:

```text
Comfort_Oriented
Balanced
Energy_Saver
Cost_Sensitive
```

Each profile represents different scheduling priorities.

### Comfort-Oriented

Higher emphasis on:

```text
comfort
immediate appliance availability
```

### Balanced

Balances:

```text
comfort
cost
peak management
preference
```

### Energy-Saver

Higher emphasis on:

```text
energy reduction
peak reduction
```

### Cost-Sensitive

Higher emphasis on:

```text
tariff cost
load shifting
```

A `user_flexibility` variable was also created.

Example:

```text
0.90
→ user accepts strong scheduling changes

0.20
→ user prefers immediate appliance operation
```

Location:

```text
build_component3_v2.py
```

Search:

```text
user_profile
user_flexibility
Comfort_Oriented
Cost_Sensitive
```

---

# 25. Why Personalization Was Important

Without personalization:

```text
same appliance
+
same load
+
same tariff
→ same recommendation for everyone
```

With personalization:

```text
same appliance
+
same load
+
same tariff

Comfort user
→ may prefer immediate operation

Cost-sensitive user
→ may prefer shifting

Energy-saver
→ may prefer ECO / load reduction
```

Personalization is therefore a central research contribution.

---

# 26. V2 Appliance Constraints

Controlled operating constraints were defined for each appliance category.

Example concepts:

### Refrigeration

```text
Critical
High priority
Maximum OFF duration ≈ 45 minutes
```

### Washing Machine

```text
Flexible
Non-interruptible while running
Maximum delay ≈ 180 minutes
```

### Dishwasher

```text
Flexible
Non-interruptible while running
Maximum delay ≈ 240 minutes
```

### Kettle

```text
High-power short-use appliance
Low delay tolerance
```

These are **research constraints**.

They are not manufacturer-certified limits.

---

# 27. V2 Target Label Generation

Supervised learning requires a target action.

Public energy datasets do not contain correct appliance scheduling decisions.

Therefore V2 uses multi-objective utility-based labels.

For every record, utility scores were calculated:

```text
utility_TURN_ON
utility_TURN_OFF
utility_DELAY
utility_ECO_MODE
utility_PRIORITY
utility_LOAD_BALANCE
```

The utilities consider:

```text
Comfort
Cost
Peak reduction
User preference
Appliance state
Criticality
Flexibility
Operating constraints
Predicted load
```

Then:

```text
Highest utility
      ↓
target_action
```

Location:

```text
build_component3_v2.py
```

Search:

```text
utility_
target_action
winning_utility
decision_margin
```

---

# 28. Winning Utility and Decision Margin

V2 also stores:

```text
winning_utility
decision_margin
```

These are research audit variables.

`decision_margin` indicates how clearly one action wins over the others.

Example:

```text
DELAY utility = 0.71
ECO_MODE      = 0.70
```

This is an ambiguous decision.

Example:

```text
PRIORITY = 0.90
TURN_ON  = 0.45
```

This is a clear decision.

Approximately:

```text
35.44%
```

of V2 records had:

```text
decision_margin < 0.05
```

meaning a large portion of the dataset contains genuinely difficult scheduling cases.

---

# 29. Data Provenance

V2 stores provenance fields to make the dataset transparent.

Conceptually:

```text
appliance_data_source
= REFIT

environment_source
= UCI / Appliances Energy contextual augmentation

forecast_source
= PP1 history-based load proxy

occupancy_source
= PP1 controlled scenario

user_profile_source
= PP1 personalization scenario

tariff_source
= PP1 temporary tariff bands

label_source
= multi-objective utility policy
```

This makes it clear which values are measured and which are derived.

---

# 30. Data Leakage Prevention

Data leakage occurs when the model receives information that reveals the target.

For example:

```text
target_action = action with highest utility
```

If the model receives:

```text
utility_TURN_ON
utility_DELAY
winning_utility
decision_margin
```

it could indirectly read the answer.

Therefore the following were excluded from model training:

```text
target_action
target_action_id
utility_*
winning_utility
decision_margin
```

Location:

```text
Component3_Processed/train_component3_v2_models.py
```

The purpose was to ensure realistic evaluation.

---

# 31. Household-Aware Data Split

The data was split by household rather than random rows.

This avoids placing very similar records from the same household into both training and test data.

### Training

```text
Houses:
5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 18, 21

Records:
150,974
```

### Validation

```text
Houses:
3, 4, 8, 20

Records:
54,189
```

### Test

```text
Houses:
1, 2, 17, 19

Records:
44,837
```

The four test households were completely unseen during model training.

This provides a stronger generalization test.

---

# 32. Models Trained

Three models were trained:

```text
Decision Tree
Random Forest
XGBoost
```

Main script:

```text
train_component3_v2_models.py
```

Approximate final results:

| Model         | Test Accuracy | Macro F1 | Model Size |
| ------------- | ------------: | -------: | ---------: |
| Decision Tree |        81.92% |   81.40% |    0.94 MB |
| Random Forest |        84.26% |   83.86% |     499 MB |
| XGBoost       |        84.68% |   84.20% |   13.86 MB |

---

# 33. Why XGBoost Was Selected

Random Forest was originally important in the proposal.

However, the final experimental results showed:

```text
XGBoost
→ slightly higher performance
→ much smaller model size
```

Random Forest:

```text
~499 MB
```

XGBoost:

```text
~13.86 MB
```

Therefore:

```text
XGBoost + Personalization
```

was selected as the final model.

---

# 34. Ambiguous vs Clear Decision Performance

The selected XGBoost model was also evaluated separately on difficult and easy decisions.

Approximate results:

```text
Ambiguous Macro F1 ≈ 0.648
Clear Macro F1     ≈ 0.949
```

This means:

- the model performs very strongly when the scheduling decision is clear
- most errors occur when several actions are almost equally reasonable

This is a useful research finding because it explains why the final accuracy is around 84–85% rather than artificially close to 100%.

---

# 35. Ablation Study

An ablation study was performed to answer:

> Which categories of information actually improve scheduling performance?

Main scripts:

```text
run_component3_ablation_study.py
run_component3_ablation_study_final.py
```

Main output:

```text
ablation_final_output/results/final_ablation_results.csv
```

---

# 36. What Is a Feature Group?

A feature group is a collection of related input columns that serve the same research purpose.

The final ablation groups were:

```text
A_Basic
B_History
C_Environment
D_Context
E_Personalization
F_Full_Interactions
```

---

# 37. Basic Feature Group

The Basic group answers:

> What is happening now?

Example features include:

```text
appliance_category
appliance_active
appliance_mean_w
current household load
hour
time information
critical_flag
flexible_flag
high_power_short_use
```

Total raw features:

```text
15
```

Result:

```text
Macro F1 = 0.5462
```

This provided the baseline.

---

# 38. History Feature Group

The History group answers:

> What happened recently?

Example features:

```text
load_lag_1
load_lag_2
load_lag_4
rolling_load_mean_1h
rolling_load_max_1h
previous_appliance_active
state_duration_minutes
on_duration_minutes
off_duration_minutes
predicted_load_w
predicted_load_level
```

Total raw features:

```text
27
```

Result:

```text
Macro F1 = 0.6465
```

Improvement:

```text
+0.1003
```

approximately:

```text
+10 percentage points
```

This shows temporal history was genuinely useful.

---

# 39. Environment Feature Group

The Environment group answers:

> What are the surrounding conditions?

Example features:

```text
temperature
humidity
environmental context
```

Total raw features:

```text
31
```

Result:

```text
Macro F1 = 0.6456
```

Increment:

```text
-0.0009
```

Therefore environment did not significantly improve the current model.

This is still a valid research finding.

It may become more useful later when live synchronized environmental sensors are integrated.

---

# 40. Context Feature Group

The Context group answers:

> What are the operating conditions?

Example features:

```text
occupancy
tariff_band
tariff_cost_index
interruptible
priority_level
max_delay_minutes
max_off_minutes
operating constraints
```

Total raw features:

```text
39
```

Result:

```text
Macro F1 = 0.6484
```

Increment:

```text
+0.0028
```

Context gave a small additional improvement.

---

# 41. Personalization Feature Group

The Personalization group answers:

> What does this user prefer?

Main features:

```text
user_profile
user_flexibility
```

Total raw features:

```text
41
```

Result:

```text
Macro F1 = 0.8427
```

Increment from context:

```text
+0.1942
```

approximately:

```text
+19.4 percentage points
```

Compared with basic:

```text
0.5462
→
0.8427
```

approximately:

```text
+29.6 percentage points
```

This was the largest improvement in the ablation study.

Therefore the ablation study strongly supports the research contribution:

```text
Personalized Appliance Scheduling
```

---

# 42. Interaction Feature Group

The final ablation experiment added engineered interaction variables.

Examples:

```text
tariff_load_interaction
flexibility_tariff_interaction
flexibility_load_interaction
load_power_interaction
off_duration_ratio
delay_duration_ratio
occupancy_comfort_interaction
critical_off_risk
flexible_shift_opportunity
peak_pressure_score
```

Total raw features:

```text
51
```

Result:

```text
Macro F1 = 0.8417
```

Change:

```text
-0.0010
```

The manually engineered interactions did not improve performance.

The likely reason is that XGBoost already learns nonlinear feature interactions internally.

Therefore:

```text
E_Personalization
```

remained the best configuration.

---

# 43. Final Ablation Results

| Experiment          | Raw Features |   Macro F1 | Incremental Gain |
| ------------------- | -----------: | ---------: | ---------------: |
| A_Basic             |           15 |     0.5462 |           0.0000 |
| B_History           |           27 |     0.6465 |          +0.1003 |
| C_Environment       |           31 |     0.6456 |          -0.0009 |
| D_Context           |           39 |     0.6484 |          +0.0028 |
| E_Personalization   |           41 | **0.8427** |      **+0.1942** |
| F_Full_Interactions |           51 |     0.8417 |          -0.0010 |

Main conclusion:

> Personalization was the strongest useful feature group.

---

# 44. Why Macro F1 Was Important

The six scheduling classes are not perfectly balanced.

Accuracy alone can hide weak performance on minority classes.

Macro F1:

1. calculates F1 separately for every class
2. gives each class equal importance
3. averages the values

Therefore Macro F1 was an important evaluation metric for this multi-class scheduling problem.

---

# 45. Rule-Based Safety Layer

Machine learning selects an optimized action.

However:

```text
Good optimization decision
≠
Always safe operational decision
```

Therefore a deterministic rule-based safety layer was added.

Flow:

```text
ML Recommendation
      ↓
Safety Validation
      ↓
Final Safe Action
```

Main file:

```text
evaluate_component3_safety_layer.py
```

Dashboard implementation:

```text
component3_dashboard.py
```

---

# 46. Safety Rule S1 – Critical Maximum OFF

If refrigeration has remained OFF beyond the configured maximum duration:

```text
do not keep the refrigerator OFF
```

Example configured PP1 value:

```text
45 minutes
```

Possible override:

```text
TURN_OFF
→
PRIORITY
```

---

# 47. Safety Rule S2 – Critical Appliance Delay

Refrigeration is treated as a critical appliance.

An inappropriate:

```text
DELAY
```

action can therefore be blocked.

---

# 48. Safety Rule S3 – Non-Interruptible Cycle

Washing machines and dishwashers are treated as non-interruptible while actively running.

Therefore:

```text
active cycle
+
TURN_OFF / DELAY
```

can be blocked.

Possible safe correction:

```text
TURN_ON
```

---

# 49. Safety Rule S4 – Maximum Delay

Flexible appliances should not remain delayed indefinitely.

If configured maximum delay is exceeded:

```text
DELAY
```

can be overridden so operation can resume.

---

# 50. Natural Safety Evaluation

Safety was evaluated on:

```text
44,837
```

unseen test records.

Results:

```text
Unsafe ML recommendations:
5,442

Unsafe rate before:
12.14%

Safety compliance before:
87.86%

Safety overrides:
5,442

Unsafe actions after:
0

Final compliance:
100%
```

Rule activation counts:

```text
S4_MAX_DELAY_EXCEEDED          5,309
S3_NON_INTERRUPTIBLE_CYCLE        97
S1_CRITICAL_MAX_OFF_EXCEEDED      36
```

---

# 51. Controlled Safety Stress Test

Natural data may not contain enough unsafe edge cases.

Therefore a controlled stress test was created.

Total scenarios:

```text
12,000
```

Main scenarios:

```text
Critical_Max_Off
Maximum_Delay_Exceeded
Running_Dishwasher
Running_Washing_Machine
```

Results:

```text
Unsafe before:
2,884

Unsafe after:
0

Final compliance:
100%
```

Important interpretation:

> 100% means compliance against the implemented experimental safety rules.

It does not mean manufacturer-certified appliance safety.

---

# 52. Why Accuracy Decreased After Safety Adjustment

Before safety:

```text
classification accuracy ≈ 84.7%
```

After safety adjustment:

```text
classification accuracy ≈ 75.2%
```

This does not mean the safety layer failed.

The reason is:

```text
ML target
= optimization policy

Safety layer
= hard operating protection
```

The safety layer intentionally changes some model decisions.

Therefore the correct safety metric is:

```text
Safety compliance
```

not post-override classification accuracy.

---

# 53. Operational Benefit Evaluation

A strong model should provide practical energy-management benefits, not only good classification metrics.

Therefore an operational simulation was implemented.

Main file:

```text
evaluate_component3_operational_benefits.py
```

Output:

```text
operational_benefit_output/
```

Evaluated measures include:

```text
energy shifted
energy avoided
ECO-mode reduction
normalized tariff cost
peak-period energy
comfort score
```

---

# 54. Operational Benefit Results

Test records:

```text
44,837
```

Energy results:

```text
Baseline observed energy:
1555.26 kWh

Scheduled current-interval energy:
1285.65 kWh

Energy shifted:
169.26 kWh

Estimated avoided interval energy:
23.65 kWh

Estimated ECO-mode reduction:
76.71 kWh
```

Cost result:

```text
Baseline normalized cost index:
1049.13

Scheduled normalized cost index:
917.88

Estimated cost-index reduction:
12.51%
```

Peak result:

```text
Baseline peak-period energy:
398.21 kWh

Scheduled peak-period energy:
328.59 kWh

Peak-period reduction:
17.48%
```

Comfort result:

```text
Average comfort score:
89.96 / 100
```

---

# 55. Important Operational Benefit Limitation

These results are:

```text
simulation-based PP1 estimates
```

They are not:

```text
measured household electricity-bill savings
```

The tariff values are normalized experimental indices.

This limitation should always be stated clearly.

---

# 56. Explainability

XGBoost is a complex model.

Therefore explainability analysis was performed to answer:

> Which features influence the scheduling decisions?

Main file:

```text
analyze_component3_explainability.py
```

Output:

```text
explainability_output/
```

Methods used:

```text
XGBoost feature importance
SHAP
prediction confidence analysis
```

---

# 57. Global Feature Importance

Top aggregated feature groups included:

```text
priority_level
appliance_category
critical_flag
appliance_active
user_profile
max_off_minutes
predicted_load_level
high_power_short_use
off_duration_minutes
flexible_flag
```

Top aggregated feature:

```text
priority_level
```

This indicates that appliance importance and operating priority strongly influence scheduling decisions.

---

# 58. SHAP Analysis

SHAP was used to analyze how individual transformed features affect model decisions.

Top transformed SHAP feature:

```text
appliance_category_Refrigeration
```

This indicates that refrigerator-related behavior strongly influences action predictions.

SHAP improves research transparency because the model is not evaluated only as a black box.

---

# 59. Prediction Confidence Analysis

The XGBoost model outputs probabilities for all scheduling actions.

Confidence bands showed the following relationship with accuracy:

| Confidence Band | Records | Accuracy |
| --------------- | ------: | -------: |
| Low             |   4,590 |   53.14% |
| Medium          |   6,350 |   64.50% |
| High            |   8,908 |   80.93% |
| Very High       |  24,989 |   97.04% |

This shows that confidence is strongly associated with prediction correctness.

---

# 60. Why the 60% Confidence Gate Was Added

A real appliance controller should not blindly execute a low-confidence prediction.

Example:

```text
TURN_OFF = 42.18%
TURN_ON  = 40.19%
```

Technically TURN_OFF has the highest probability.

However, the model is highly uncertain.

Therefore:

```text
CONFIDENCE_THRESHOLD = 60%
```

was introduced.

Logic:

```text
Confidence >= 60%
→ accept ML recommendation

Confidence < 60%
→ do not directly execute the state change
→ preserve current appliance state
```

The result is then still checked by the safety layer.

---

# 61. Confidence Gate Location

Main implementation:

```text
component3_dashboard.py
```

Useful search terms:

```text
CONFIDENCE_THRESHOLD
apply_confidence_gate
```

---

# 62. Confidence Gate vs Safety Layer

These mechanisms solve different problems.

## Confidence Gate

Handles:

```text
ML uncertainty
```

Question:

> Is the model confident enough to automatically act?

## Safety Layer

Handles:

```text
hard operating constraints
```

Question:

> Even if the model is confident, is the action operationally allowed?

Final flow:

```text
XGBoost
  ↓
Confidence Gate
  ↓
Safety Layer
  ↓
Final Safe Action
```

---

# 63. Final Streamlit Dashboard

The final integrated PP1 dashboard is implemented in:

```text
component3_dashboard.py
```

Run using:

```bash
python -m streamlit run component3_dashboard.py
```

Default local address:

```text
http://localhost:8501
```

---

# 64. Dashboard Inputs

The dashboard accepts:

```text
Appliance
User Profile
Tariff Period
Appliance Currently ON/OFF
Home Occupancy
User Flexibility
Current Appliance Power
Predicted Household Load
Current OFF Duration
Indoor Temperature
Indoor Humidity
Hour of Day
```

---

# 65. Dashboard Outputs

The dashboard displays:

```text
ML Recommendation
Prediction Confidence
Confidence-Gated Action
Final Safe Action
Confidence Validation
Safety Validation
Decision Pipeline
Prediction Probabilities
Decision Context
Global Feature Importance
MQTT Command Preview
Research Validation Summary
```

---

# 66. Dashboard Runtime Flow

```text
User / Appliance Context
         ↓
XGBoost Model
         ↓
Predicted Action
         ↓
Prediction Probability
         ↓
60% Confidence Gate
         ↓
Candidate Action
         ↓
Rule-Based Safety Layer
         ↓
Final Safe Action
         ↓
MQTT-Ready Command
```

---

# 67. MQTT-Ready Output

The dashboard creates an MQTT-ready topic.

Example:

```text
smart_home/component3/refrigeration/command
```

Example JSON payload:

```json
{
  "component": "component_3",
  "appliance": "Refrigeration",
  "ml_recommendation": "TURN_OFF",
  "confidence": 0.4218,
  "confidence_threshold": 0.6,
  "confidence_gate_triggered": true,
  "confidence_gate_reason": "LOW_CONFIDENCE_HOLD_CURRENT_STATE",
  "gated_action": "TURN_ON",
  "safety_override": false,
  "safety_rule": "NONE",
  "final_action": "TURN_ON"
}
```

---

# 68. Relay Mapping

The PP1 hardware mapping is:

```text
Refrigeration
→ Relay Channel 1

Washing Machine
→ Relay Channel 2

Dishwasher
→ Relay Channel 3

Kettle
→ Relay Channel 4
```

Actual real MQTT publishing and ESP32 control remain future implementation tasks.

---

# 69. PP1 Demo Scenario 1 – Cost-Sensitive Kettle During Peak Tariff

Inputs:

```text
Appliance: Kettle
User Profile: Cost_Sensitive
Tariff Period: Peak
Appliance Currently ON: Yes
Home Occupied: Yes
User Flexibility: 0.90
Current Appliance Power: 1800 W
Predicted Household Load: 4200 W
Current OFF Duration: 0 min
Indoor Temperature: 24 C
Humidity: 50%
Hour: 19
```

Purpose:

```text
Tariff-aware / high-load scheduling
```

Explanation:

```text
Cost-sensitive user
+
Peak tariff
+
High predicted load
+
High-power kettle
```

creates strong pressure to reduce or shift current power demand.

The exact ML action may vary based on the complete 41-feature context.

Possible energy-management actions include:

```text
LOAD_BALANCE
ECO_MODE
TURN_OFF
```

Panel explanation:

> Peak tariff, high demand and a cost-sensitive user give the system a strong reason to reduce current energy pressure.

---

# 70. PP1 Demo Scenario 2 – Comfort-Oriented Kettle During Off-Peak

Inputs:

```text
Appliance: Kettle
User Profile: Comfort_Oriented
Tariff Period: Off_Peak
Appliance Currently ON: Yes
Home Occupied: Yes
User Flexibility: 0.20
Current Appliance Power: 1800 W
Predicted Household Load: 800 W
Current OFF Duration: 0 min
Indoor Temperature: 24 C
Humidity: 50%
Hour: 2
```

Purpose:

```text
Personalization
```

Explanation:

```text
Comfort-oriented user
+
Low flexibility
+
Cheap tariff
+
Low predicted load
```

means there is less reason to aggressively stop or delay appliance use.

Comparing Scenario 1 and Scenario 2 demonstrates personalization.

Panel explanation:

> The appliance is the same, but user preference, flexibility, tariff and household load are different, so scheduling behavior can also change.

---

# 71. PP1 Demo Scenario 3 – Known Low-Confidence Refrigerator

Inputs:

```text
Appliance: Refrigeration
User Profile: Balanced
Tariff Period: Day
Appliance Currently ON: Yes
Home Occupied: Yes
User Flexibility: 0.50
Current Appliance Power: 0.62 W
Predicted Household Load: 307.28 W
Current OFF Duration: 15 min
Indoor Temperature: 24 C
Humidity: 50%
Hour: 18
```

Previously observed result:

```text
ML Recommendation:
TURN_OFF

Prediction Confidence:
42.18%

Confidence Threshold:
60%

Confidence Gate:
TRIGGERED

Confidence-Gated Action:
TURN_ON

Safety Layer:
PASSED

Final Safe Action:
TURN_ON
```

Explanation:

> The ML model selected TURN_OFF, but its confidence was only 42.18%. Because the result was below the 60% threshold, the controller did not directly execute the uncertain state change. Since the appliance was already ON, the confidence gate preserved the current state as TURN_ON.

---

# 72. PP1 Demo Scenario 4 – Refrigerator OFF Too Long

Inputs:

```text
Appliance: Refrigeration
User Profile: Cost_Sensitive
Tariff Period: Peak
Appliance Currently ON: No
Home Occupied: Yes
User Flexibility: 0.90
Current Appliance Power: 0 W
Predicted Household Load: 3500 W
Current OFF Duration: 60 min
Indoor Temperature: 27 C
Humidity: 60%
Hour: 19
```

Configured refrigerator maximum OFF duration:

```text
45 minutes
```

Purpose:

```text
Critical safety protection
```

If the candidate action attempts to keep the refrigerator OFF:

```text
S1_CRITICAL_MAX_OFF_EXCEEDED
```

can be activated.

Possible safe correction:

```text
PRIORITY
```

Explanation:

> Cost and peak-load optimization are important, but critical refrigerator operating constraints have higher priority.

---

# 73. PP1 Demo Scenario 5 – Running Washing Machine During Peak

Inputs:

```text
Appliance: Washing Machine
User Profile: Cost_Sensitive
Tariff Period: Peak
Appliance Currently ON: Yes
Home Occupied: Yes
User Flexibility: 0.90
Current Appliance Power: 1000 W
Predicted Household Load: 4500 W
Current OFF Duration: 0 min
Indoor Temperature: 25 C
Humidity: 55%
Hour: 20
```

Purpose:

```text
Non-interruptible cycle protection
```

The optimizer may have strong reasons to reduce current demand because:

```text
Peak tariff
+
High household load
+
Cost-sensitive user
+
High flexibility
```

However, if the model/gate produces:

```text
TURN_OFF
or
DELAY
```

while the washing machine is actively running:

```text
S3_NON_INTERRUPTIBLE_CYCLE
```

can override the action.

Possible safe action:

```text
TURN_ON
```

Explanation:

> Energy optimization cannot interrupt an active non-interruptible appliance cycle.

---

# 74. Recommended Demo Story

The five demo scenarios can be presented as:

```text
Scenario 1
→ Cost / peak-aware scheduling

Scenario 2
→ Personalization

Scenario 3
→ Low-confidence AI protection

Scenario 4
→ Critical refrigerator safety

Scenario 5
→ Active appliance cycle protection
```

The complete story is:

> The model optimizes energy use, considers user preferences, avoids blindly trusting uncertain AI, and applies deterministic safety rules before generating the final controller command.

---

# 75. Python Environment Setup

Open:

```text
Component3_Processed
```

in the VS Code terminal.

Create virtual environment:

```bash
python -m venv .venv
```

Activate on Windows:

```bash
.venv\Scripts\activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 76. Important Model Compatibility

The saved scikit-learn pipeline was created using:

```text
scikit-learn==1.3.2
```

Verify:

```bash
python -c "import sklearn; print(sklearn.__version__)"
```

Expected:

```text
1.3.2
```

Do not casually upgrade scikit-learn before PP1.

Serialized scikit-learn pipelines can become incompatible across versions.

---

# 77. Scientific Python Compatibility

The frozen model environment should remain compatible with:

```text
numpy==1.26.4
scikit-learn==1.3.2
```

If a newer PyArrow package complains that NumPy 2.x is required, keep NumPy 1.26.4 for model compatibility and use a compatible PyArrow release instead of upgrading NumPy without testing.

For example:

```text
pyarrow==14.0.2
```

can be used with the NumPy 1.26.x environment.

---

# 78. Start the Dashboard

Run:

```bash
python -m streamlit run component3_dashboard.py
```

Then open:

```text
http://localhost:8501
```

---

# 79. Do Not Retrain for the PP1 Demo

Fernando does not need to rebuild the research before demonstrating PP1.

Do not immediately rerun:

```text
prepare_component3_base_dataset.py
prepare_feature_fusion.py
generate_scheduling_labels.py
build_component3_v2.py
train_component3_models.py
train_component3_v2_models.py
run_component3_ablation_study_final.py
```

These scripts can recreate datasets, models and research evidence.

For the PP1 demo, use the frozen artifacts.

Important folders:

```text
v2_output/
v2_model_output/
ablation_final_output/
safety_layer_output/
operational_benefit_output/
explainability_output/
```

---

# 80. Important Scripts and Their Purposes

| Script                                        | Purpose                                       |
| --------------------------------------------- | --------------------------------------------- |
| `audit_refit.py`                              | Audit REFIT source files                      |
| `build_appliance_summary.py`                  | Summarize appliances and households           |
| `prepare_component3_base_dataset.py`          | Create 15-minute REFIT-based scheduling data  |
| `prepare_feature_fusion.py`                   | Add environment/context features              |
| `generate_scheduling_labels.py`               | Generate V1 deterministic labels              |
| `train_component3_models.py`                  | Train V1 baseline models                      |
| `build_component3_v2.py`                      | Build advanced personalized V2 dataset        |
| `train_component3_v2_models.py`               | Train DT / RF / XGBoost on V2                 |
| `run_component3_ablation_study.py`            | Initial ablation analysis                     |
| `run_component3_ablation_study_final.py`      | Final ablation experiment                     |
| `evaluate_component3_safety_layer.py`         | Safety rules + natural/stress evaluation      |
| `evaluate_component3_operational_benefits.py` | Simulated energy/cost/peak/comfort evaluation |
| `analyze_component3_explainability.py`        | Feature importance + SHAP + confidence        |
| `component3_dashboard.py`                     | Final integrated PP1 Streamlit dashboard      |

---

# 81. Important Output Folders

## `v2_output/`

Contains:

```text
final V2 dataset
```

## `v2_model_output/`

Contains:

```text
model evaluation results
```

## `ablation_final_output/`

Contains:

```text
final ablation results
plots
models
```

## `safety_layer_output/`

Contains:

```text
natural safety evaluation
stress test outputs
```

## `operational_benefit_output/`

Contains:

```text
energy simulation
cost simulation
peak analysis
comfort analysis
```

## `explainability_output/`

Contains:

```text
feature importance
SHAP analysis
confidence results
plots
```

---

# 82. Main Results Summary

## V2 Dataset

```text
250,000 records
20 households
6 scheduling classes
```

## Selected Model

```text
XGBoost + Personalization
```

## Final Model Performance

```text
Accuracy ≈ 84.7%
Macro F1 ≈ 84.2%
```

## Ablation

```text
Basic             ≈ 54.6%
+ History         ≈ 64.6%
+ Environment     ≈ 64.6%
+ Context         ≈ 64.8%
+ Personalization ≈ 84.3%
+ Interactions    ≈ 84.2%
```

## Natural Safety Evaluation

```text
44,837 unseen records
5,442 unsafe before
0 unsafe after
100% rule compliance
```

## Safety Stress Test

```text
12,000 scenarios
2,884 unsafe before
0 unsafe after
```

## Operational Simulation

```text
Cost-index reduction ≈ 12.51%
Peak reduction ≈ 17.48%
Comfort ≈ 89.96 / 100
```

## Confidence

```text
Low       ≈ 53.14% accuracy
Medium    ≈ 64.50%
High      ≈ 80.93%
Very High ≈ 97.04%
```

---

# 83. Main Limitations

The current PP1 implementation has known limitations.

## No Perfect Dataset

No single public dataset contained:

```text
appliance
environment
occupancy
tariff
user preference
forecast
scheduling action
```

together.

---

## Environmental Data

Environmental values come from a different dataset.

They are used as contextual augmentation.

They are not synchronized REFIT household measurements.

---

## Occupancy

Occupancy is currently scenario-generated.

Future:

```text
PIR / live sensing
```

---

## Predicted Load

The predicted load is currently a history-based PP1 proxy.

Future:

```text
Component 2 real forecast
```

---

## Tariff

Current tariff values are normalized research indices.

They are not official electricity prices.

---

## User Profiles

Current profiles are controlled research personas.

Future versions should learn from:

```text
user configuration
manual overrides
historical appliance behavior
```

---

## Target Labels

V2 target actions are utility/policy-derived.

They are not historical human scheduling decisions.

---

## Appliance Coverage

Current PP1 uses four representative appliance categories.

---

## Safety

Safety rules are research constraints.

They are not manufacturer-certified safety limits.

---

## Operational Savings

Cost/peak improvements are simulation-based estimates.

They are not measured electricity-bill savings.

---

# 84. Main Challenges Faced

Important implementation challenges included:

```text
Large REFIT files
House 1 incomplete working copy
No single complete dataset
Environmental source mismatch
No natural scheduling labels
V1 near-100% accuracy
Feature leakage risk
Multi-class scheduling
Unseen-household generalization
Unsafe ML recommendations
Low-confidence predictions
Python package/model compatibility
```

Each problem was addressed explicitly during the research implementation.

---

# 85. What Is Completed for PP1

The following work is complete:

```text
Research scope analysis

REFIT dataset investigation
REFIT audit
House 1 correction
Appliance mapping
Appliance category selection

8-second → 15-minute processing
Base dataset creation

Environmental contextual augmentation

Occupancy scenario
Tariff scenario
User-profile scenario
User flexibility
Predicted-load proxy

V1 dataset
V1 scheduling labels
V1 model comparison
Identification of V1 limitation

V2 advanced dataset
History / lag features
Rolling features
Appliance constraints
Personalization
Multi-objective utility labels
Decision-margin analysis
Data provenance
Leakage prevention

Household-aware split
Decision Tree
Random Forest
XGBoost
Model comparison
XGBoost selection

Ablation study
Personalization contribution validation

Rule-based safety layer
Natural safety evaluation
12,000-case stress test

Operational benefit evaluation
Energy analysis
Peak analysis
Normalized cost analysis
Comfort analysis

Feature importance
SHAP
Prediction-confidence analysis

60% confidence gate

Integrated Streamlit dashboard

MQTT-ready topic
MQTT-ready JSON payload

Environment / requirements setup
Technical handover documentation
```

---

# 86. What Remains for PP2 / Final

The next major stage is:

```text
Streamlit
   ↓
Real MQTT Broker
   ↓
ESP32
   ↓
4-Channel Relay
   ↓
Safe Representative Load
```

Future work includes:

```text
Real MQTT publish
Python MQTT subscriber test
ESP32 subscriber
Relay control
Component 2 forecast integration
Live occupancy integration
Final tariff integration
Hardware response-time measurement
Actual energy measurement
Real user feedback
Preference learning
More appliance categories
Final end-to-end system validation
```

---

# 87. Final Architecture

```text
Appliance State
User Profile
User Flexibility
Tariff
Occupancy
Environment
Predicted Load
        │
        ▼
┌──────────────────────┐
│ XGBoost Personalized │
│ Scheduling Model     │
└──────────┬───────────┘
           │
           ▼
Prediction + Probability
           │
           ▼
┌──────────────────────┐
│ 60% Confidence Gate  │
└──────────┬───────────┘
           │
           ▼
Candidate Action
           │
           ▼
┌──────────────────────┐
│ Rule-Based Safety    │
│ Layer                │
└──────────┬───────────┘
           │
           ▼
Final Safe Action
           │
           ▼
MQTT-Ready JSON
           │
           ▼
Future:
MQTT Broker
           │
           ▼
ESP32
           │
           ▼
Relay
           │
           ▼
Appliance
```

---

# 88. Full Component 3 Development Story

```text
REFIT
↓
Audit household/appliance data
↓
Correct House 1 working-data issue
↓
Map Appliance1...Appliance9
↓
Remove issue records
↓
Select 4 representative appliance categories
↓
Convert ~8-second readings to 15-minute intervals
↓
Create REFIT scheduling base dataset
↓
Add environmental context from second dataset
↓
Add PP1 occupancy / tariff / user / forecast context
↓
Create V1 deterministic scheduling labels
↓
Train DT / RF / XGB
↓
Detect unrealistic near-100% result
↓
Redesign the research problem
↓
Build V2
↓
Add history / lags / rolling features
↓
Add predicted-load proxy
↓
Add personalization
↓
Add appliance constraints
↓
Create multi-objective utility targets
↓
Prevent leakage
↓
Split by household
↓
Train DT / RF / XGBoost
↓
Select XGBoost
↓
Run ablation study
↓
Prove personalization is strongest contribution
↓
Add rule-based safety layer
↓
Natural safety evaluation
↓
12,000-case stress test
↓
Operational benefit simulation
↓
Feature importance
↓
SHAP explainability
↓
Confidence analysis
↓
Add 60% confidence gate
↓
Build Streamlit dashboard
↓
Generate MQTT-ready payload
↓
NEXT:
Real MQTT + ESP32 + relay
```

---

# 89. PP1 One-Minute Explanation

Component 3 is the personalized appliance scheduling decision layer of the smart energy management system.

REFIT was selected because it provides real appliance-level electrical behavior, but it did not contain all the environmental, occupancy, tariff, user preference, future-load and scheduling-action information required by Component 3.

Therefore REFIT was used as the main electrical source, while a second Appliances Energy Prediction dataset was used only for realistic environmental context. Other missing PP1 variables such as occupancy, user profiles, tariff and predicted load were transparently created using controlled scenarios or derived features.

REFIT's high-frequency measurements were converted into 15-minute scheduling intervals.

The first V1 dataset used direct IF/ELSE scheduling rules and produced almost 100% accuracy because the target rules were too directly represented by the inputs. Therefore V1 was retained only as a baseline and the problem was redesigned into V2.

V2 introduced temporal history, a history-based predicted-load proxy, user profiles, flexibility, appliance constraints and multi-objective utility-based target actions. Leakage-sensitive utility variables were excluded from training, and households were separated between training, validation and testing.

Decision Tree, Random Forest and XGBoost were compared. XGBoost was selected with approximately 84.7% accuracy and 84.2% Macro F1.

A group-wise ablation study showed that personalization was the strongest feature contribution, increasing Macro F1 from approximately 64.8% to 84.3%.

A deterministic safety layer then corrected unsafe actions and achieved 100% compliance against the implemented research rules on both unseen test data and a 12,000-case safety stress test.

Operational simulation estimated approximately 12.5% normalized cost-index reduction, 17.5% peak exposure reduction and around 90/100 user comfort.

Feature importance and SHAP were used for explainability.

Prediction-confidence analysis showed that low-confidence recommendations were much less reliable, so a 60% confidence gate was added before the safety layer.

The final Streamlit dashboard demonstrates the complete PP1 software decision pipeline and produces an MQTT-ready command.

The next stage is real MQTT communication, ESP32 integration and relay actuation.

---

# 90. Quick Setup Commands

Create environment:

```bash
python -m venv .venv
```

Activate:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Verify scikit-learn:

```bash
python -c "import sklearn; print(sklearn.__version__)"
```

Expected:

```text
1.3.2
```

Run dashboard:

```bash
python -m streamlit run component3_dashboard.py
```

---

# 91. Important PP1 Handover Rule

For the PP1 demo:

> Do not rebuild or retrain the entire model pipeline unless there is a deliberate research change.

Use the already-generated research artifacts and selected XGBoost model.

The main demonstration should focus on:

```text
Input Context
↓
XGBoost Recommendation
↓
Prediction Confidence
↓
60% Confidence Gate
↓
Safety Layer
↓
Final Safe Action
↓
MQTT-Ready Command
```

---

# 92. Final PP1 Status

**Component 3 PP1 software implementation is substantially complete.**

Completed:

```text
Dataset investigation
Dataset engineering
V1 baseline
V2 advanced dataset
Model comparison
Model selection
Ablation study
Safety validation
Stress testing
Operational simulation
Explainability
Confidence analysis
Confidence gate
Streamlit dashboard
MQTT-ready output
```

Remaining major implementation:

```text
Real MQTT communication
ESP32 integration
Relay actuation
Component 2 live forecast
Live sensor/context integration
Final hardware validation
```

---

# 93. Final Summary

The Component 3 PP1 work demonstrates a complete personalized appliance scheduling decision engine.

The main contribution is not only model accuracy.

The complete contribution includes:

```text
Real appliance behavior
+
Temporal history
+
Personalization
+
Multi-objective scheduling
+
Household-aware generalization
+
Leakage prevention
+
Model comparison
+
Ablation validation
+
Safety protection
+
Operational benefit simulation
+
Explainability
+
Confidence-aware control
+
Dashboard demonstration
+
MQTT-ready integration
```

The result is a research prototype that can make context-aware personalized appliance decisions while preventing low-confidence and operationally invalid actions from directly reaching the future hardware controller.
