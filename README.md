###### J26-IT-389_Smart_Energy_MS_RP

# AI-Based Smart Energy Management System using IoT, Edge Computing and Machine Learning

**Research Project ID:** J26-IT-389
|
**Module:** IT4010 – Research Project
---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Research Problem](#2-research-problem)
3. [Research Objectives](#3-research-objectives)
4. [Proposed System Architecture](#4-proposed-system-architecture)
5. [Research Components](#5-research-components)
6. [Technology Stack](#6-technology-stack)
7. [System Workflow](#7-system-workflow)
8. [Dataset Sources](#8-dataset-sources)
9. [Model Evaluation and Performance Metrics](#9-model-evaluation-and-performance-metrics)
10. [Repository Structure](#10-repository-structure)
11. [Installation and Setup](#11-installation-and-setup)
12. [Running the Prototype](#12-running-the-prototype)
13. [Expected System Capabilities](#13-expected-system-capabilities)
14. [Research Novelty and Contributions](#14-research-novelty-and-contributions)
15. [Current Implementation Status](#15-current-implementation-status)
16. [Limitations and Future Improvements](#16-limitations-and-future-improvements)
17. [Research Team](#17-research-team)
18. [References](#18-references)

---

## 1. Project Overview

The **AI-Based Smart Energy Management System using IoT, Edge Computing and Machine Learning** is a research project focused on improving household energy monitoring, electricity consumption forecasting, appliance scheduling, and energy optimization through intelligent computing technologies.

Traditional energy monitoring systems primarily measure electricity consumption and support billing. However, they often lack integrated environmental monitoring, intelligent anomaly detection, multi-horizon forecasting, personalized appliance control, and continuous model adaptation.

This research proposes an integrated Smart Home Energy Management System (SHEMS) that combines Internet of Things (IoT) sensors, edge computing, MQTT communication, statistical forecasting, machine learning, deep learning, and intelligent decision-making.

The proposed system collects electricity consumption and environmental information, processes sensor data, predicts future energy demand, generates appliance scheduling decisions, and uses feedback to improve future predictions.

The overall research is organized into four components:

- **Component 1:** Intelligent Edge Data Processing and Anomaly Detection
- **Component 2:** Hybrid Multi-Horizon Energy Consumption Prediction
- **Component 3:** Intelligent Appliance Scheduling and Energy Optimization
- **Component 4:** Adaptive Edge Intelligence and Federated Learning

Each component addresses a distinct research problem while contributing to the overall objective of developing an intelligent and adaptive energy management system.

### Main Research Goal

To develop an AI-based smart energy management system that integrates IoT sensors, edge computing, and hybrid machine learning models for real-time energy monitoring, prediction, and optimization.

---

## 2. Research Problem

Electricity consumption is increasing in residential and industrial environments, creating challenges in efficient monitoring, prediction, and management.

Conventional energy monitoring systems primarily focus on collecting electricity measurements and generating bills. Although IoT-based smart meters improve remote monitoring, many existing approaches rely mainly on energy consumption data and do not adequately incorporate environmental conditions or occupancy information.

Other important limitations include:

- Limited intelligent preprocessing of raw sensor data.
- Insufficient detection of abnormal energy readings and faulty sensor measurements.
- Dependence on individual forecasting models rather than complementary prediction techniques.
- Limited integration of prediction results with automated appliance scheduling.
- Inadequate consideration of occupancy, user preferences, electricity tariffs, and environmental comfort.
- Dependence on centralized processing, which may introduce communication overhead and response delays.
- Limited adaptation to changing energy consumption patterns over time.
- Privacy concerns when raw household data must be transferred to centralized servers.

This research addresses these challenges through a multi-component architecture that combines edge data processing, hybrid forecasting, intelligent appliance scheduling, and adaptive machine learning.

---

## 3. Research Objectives

### 3.1 Main Objective

To develop an AI-based smart energy management system that integrates IoT sensors, edge computing, and hybrid machine learning models for real-time energy monitoring, prediction, and optimization.

### 3.2 Specific Objectives

1. Develop an intelligent edge-processing framework to collect, validate, preprocess, and detect anomalies in real-time household energy and environmental data.
2. Develop a hybrid prediction framework that forecasts household electricity consumption across multiple prediction horizons.
3. Develop an intelligent appliance scheduling and optimization framework that generates personalized control decisions using predicted energy demand, occupancy, environmental conditions, electricity tariffs, and user preferences.
4. Develop an adaptive edge intelligence framework that improves prediction performance using newly collected data while supporting privacy-preserving federated learning.

### 3.3 Overall Research Outcome

The intended outcome is a unified prototype that connects sensor data collection, intelligent preprocessing, energy forecasting, appliance scheduling, device control, and adaptive learning.

---

## 4. Proposed System Architecture

The proposed system follows a distributed architecture in which IoT devices collect sensor measurements, edge processing improves data quality, predictive models estimate future energy demand, and intelligent decision-making generates energy optimization actions.

### 4.1 Architecture Overview

```text
+--------------------------------------------------+
|                 IoT SENSOR LAYER                 |
|                                                  |
| ESP32 | PZEM-004T | DHT22 | PIR | LDR | Relay    |
+------------------------+-------------------------+
                         |
                         v
+--------------------------------------------------+
|          COMPONENT 1: EDGE PROCESSING            |
|                                                  |
| Data Validation | Preprocessing | Feature        |
| Engineering | Isolation Forest Anomaly Detection|
+------------------------+-------------------------+
                         |
                         v
+--------------------------------------------------+
|             MQTT COMMUNICATION LAYER             |
|                                                  |
| Lightweight Sensor Data and Control Messaging   |
+------------------------+-------------------------+
                         |
                         v
+--------------------------------------------------+
|          COMPONENT 2: ENERGY PREDICTION           |
|                                                  |
| EMA | ARIMA | LSTM | Multi-Horizon Forecasting   |
+------------------------+-------------------------+
                         |
                         v
+--------------------------------------------------+
|       COMPONENT 3: ENERGY OPTIMIZATION            |
|                                                  |
| Occupancy | User Preferences | Tariffs | Safety  |
| Intelligent Scheduling | Appliance Control       |
+------------------------+-------------------------+
                         |
                         v
+--------------------------------------------------+
|            APPLIANCE CONTROL LAYER               |
|                                                  |
| Relay / Smart Plug | ON/OFF | Delay | Eco Mode   |
+------------------------+-------------------------+
                         |
                         v
+--------------------------------------------------+
|          COMPONENT 4: ADAPTIVE INTELLIGENCE       |
|                                                  |
| Feedback | Model Drift | GRU | Online Learning   |
| Federated Learning | Model Updates               |
+------------------------+-------------------------+
                         |
                         v
+--------------------------------------------------+
|            CONTINUOUS MONITORING                 |
|                                                  |
| Evaluation | Feedback | Updated Predictions      |
+--------------------------------------------------+
```

*Figure: Conceptual architecture derived from the research proposal. It represents the intended system design and does not, by itself, establish that every integration has been implemented.*

### 4.2 IoT Sensor Layer

The proposed hardware layer uses an ESP32 microcontroller and the following components:

| Component | Purpose |
|---|---|
| ESP32 | Sensor data collection, edge-device processing, and communication |
| PZEM-004T | Electricity measurements, including voltage, current, power, and energy |
| DHT22 | Temperature and humidity monitoring |
| PIR sensor | Occupancy and motion-related information |
| LDR | Light-intensity measurement |
| Relay module / smart plug | Appliance switching and control |

The collected data supports energy forecasting and context-aware appliance scheduling.

### 4.3 Edge Processing Layer

Component 1 processes raw sensor data before it is used by downstream components. Its responsibilities include data validation, missing-value handling, noise removal, duplicate removal, feature scaling, feature engineering, and anomaly detection.

The proposed Isolation Forest algorithm identifies potentially abnormal energy readings and faulty sensor values.

### 4.4 Communication Layer

MQTT provides lightweight communication between IoT devices and the server or processing services.

The research also proposes evaluating communication behavior and reducing unnecessary data transmission through intelligent processing.

### 4.5 Prediction Layer

Component 2 uses a combination of statistical and deep learning techniques to forecast energy consumption across different time horizons.

### 4.6 Decision and Control Layer

Component 3 combines prediction results with occupancy, environmental conditions, electricity tariffs, and user preferences to generate appliance scheduling decisions.

A rule-based safety layer is intended to protect critical appliances before automated control actions are executed.

### 4.7 Adaptive Intelligence Layer

Component 4 focuses on adapting deployed prediction models to changing household energy usage and environmental conditions through feedback, drift detection, GRU-based learning, online learning, and federated learning.

---

## 5. Research Components

### 5.1 Component 1 — Intelligent Edge Data Processing and Anomaly Detection

**Research focus:** Reliable and efficient preprocessing of household energy and environmental data at the edge.

#### Objectives

- Collect energy and environmental data from IoT sensors.
- Validate and preprocess incoming measurements.
- Handle missing values, noise, duplicate records, and inconsistent readings.
- Perform feature engineering and prepare data for prediction.
- Apply Isolation Forest for anomaly detection.
- Evaluate the effectiveness and efficiency of edge processing.

#### Proposed Technologies

- ESP32
- PZEM-004T, DHT22, PIR, and LDR sensors
- Python
- Pandas and NumPy
- Scikit-learn
- Isolation Forest
- MQTT

#### Evaluation Metrics

- Precision
- Recall
- F1-score
- False-positive rate
- MQTT bandwidth reduction
- Edge resource utilization

#### Intended Contribution

The component aims to improve the reliability of downstream prediction by identifying abnormal data and reducing the transmission of unnecessary or unreliable sensor measurements.

---

### 5.2 Component 2 — Hybrid Multi-Horizon Energy Consumption Prediction

**Research focus:** Forecasting household electricity consumption across different prediction horizons using complementary statistical and deep learning techniques.

#### Objectives

- Collect and preprocess public and prototype energy datasets.
- Extract relevant historical, temporal, environmental, and occupancy features where available.
- Implement Exponential Moving Average (EMA) for next-hour prediction.
- Implement ARIMA for short-term forecasting.
- Implement Long Short-Term Memory (LSTM) for longer-horizon forecasting.
- Develop horizon-based model-selection logic.
- Compare prediction performance across the selected approaches.

#### Proposed Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Statsmodels
- TensorFlow / Keras
- EMA, ARIMA, and LSTM

#### Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- R² score
- Training time
- Inference time

#### Intended Contribution

The component proposes a hybrid forecasting framework that selects an appropriate prediction approach according to the forecasting horizon rather than relying exclusively on a single model.

---

### 5.3 Component 3 — Intelligent Appliance Scheduling and Energy Optimization

**Research focus:** Generating safe, personalized, and context-aware appliance scheduling decisions using predicted energy demand.

#### Objectives

- Analyze predicted energy consumption alongside occupancy, indoor environmental conditions, electricity tariffs, and user preferences.
- Implement a rule-based safety layer for critical appliances.
- Develop an intelligent decision engine for appliance scheduling.
- Compare scheduling approaches using suitable machine learning baselines.
- Generate optimized control actions.
- Communicate control commands to the ESP32 through MQTT as part of the proposed integrated system.

#### Proposed Technologies

- Python
- Scikit-learn
- Random Forest
- Decision Tree and XGBoost for comparison
- Rule-based safety logic
- MQTT
- Streamlit for the available research dashboard prototype

#### Supported Decision Types in the Proposed Design

- Turn ON / Turn OFF
- Delayed operation
- Eco mode
- Priority scheduling
- Load balancing

Actual control decisions must respect the safety rules and the requirements of critical appliances.

#### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Estimated energy savings
- User comfort score
- Decision response time
- Safety compliance

#### Intended Contribution

This component extends energy forecasting into actionable decision support by considering household context, user preferences, and appliance safety.

---

### 5.4 Component 4 — Adaptive Edge Intelligence and Federated Learning

**Research focus:** Improving prediction performance over time while reducing the need to transmit raw household data to a centralized server.

#### Objectives

- Deploy a trained prediction model in an edge environment.
- Collect feedback and newly generated sensor data.
- Detect model drift caused by changing usage patterns and environmental conditions.
- Apply GRU-based adaptive learning.
- Investigate online learning and federated learning for model updates.
- Evaluate the effectiveness of adaptation and privacy-preserving learning.

#### Proposed Technologies

- Python
- TensorFlow / Keras
- GRU
- Online learning techniques
- Federated learning
- Edge computing
- Model drift detection

#### Evaluation Considerations

- Prediction performance before and after adaptation
- MAE and RMSE
- Model update performance
- Adaptation or retraining time
- Edge resource utilization
- Communication overhead
- Data privacy characteristics

#### Intended Contribution

This component investigates how deployed models can remain useful as household energy consumption patterns change, while supporting privacy-preserving distributed learning.

---

## 6. Technology Stack

The following technologies are identified in the research proposal or associated component development. Their use and implementation status may differ between components.

| Category | Technologies |
|---|---|
| Programming language | Python |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Statistical forecasting | EMA, ARIMA |
| Deep learning | LSTM, GRU |
| Model development | TensorFlow, Keras |
| Anomaly detection | Isolation Forest |
| Appliance scheduling | Random Forest, rule-based safety logic |
| Model comparison | Decision Tree, XGBoost |
| IoT microcontroller | ESP32 |
| Energy measurement | PZEM-004T |
| Environmental sensing | DHT22, PIR, LDR |
| Device communication | MQTT |
| Appliance control | Relay module / smart plug |
| Prototype dashboard | Streamlit |
| Evaluation | MAE, RMSE, MAPE, R², Precision, Recall, F1-score |

**Note:** This table describes the research technology stack. It does not mean every technology is installed, integrated, or used in every component.

---

## 7. System Workflow

The overall research follows this workflow:

1. **Sense:** Collect electricity consumption, environmental, and occupancy-related measurements.
2. **Clean:** Validate, preprocess, and check incoming sensor data for anomalies.
3. **Transmit:** Exchange relevant data using MQTT.
4. **Predict:** Forecast energy consumption for the required prediction horizon.
5. **Decide:** Generate appliance scheduling recommendations using predictions and contextual information.
6. **Protect:** Apply safety rules before executing appliance control decisions.
7. **Control:** Send permitted control commands to the relevant device-control layer.
8. **Monitor:** Observe energy usage and evaluate system performance.
9. **Learn:** Use new data and feedback to investigate model adaptation and future prediction improvements.

**Overall workflow:**

`Sense → Clean → Predict → Decide → Control → Monitor → Learn`

The learning stage supports continuous improvement, while the safety layer remains an important condition for automated appliance control.

---

## 8. Dataset Sources

The research proposal identifies public energy and environmental datasets that can support training, benchmarking, and validation.

| Dataset / Resource | Intended Use | Source |
|---|---|---|
| Smart* data | Household energy consumption research | https://jack-kelly.com/data/ |
| REFIT Electrical Load Dataset | Residential electrical load analysis | https://pureportal.strath.ac.uk/ |
| UCI Individual Household Electric Power Consumption | Historical household electricity measurements | https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption |
| Pecan Street Dataport | Residential energy data, subject to access conditions | https://www.pecanstreet.org/dataport/ |
| NOAA Climate Data | Environmental and climate information | https://www.ncei.noaa.gov/ |
| Kaggle energy datasets | Supplementary energy datasets | https://www.kaggle.com/datasets |
| OpenWeatherMap | Weather-related information, subject to API access and terms | https://openweathermap.org/ |

### Dataset Usage Principles

- Use datasets appropriate to the research objective of each component.
- Record dataset names, source links, selected features, and preprocessing steps.
- Preserve the chronological order of time-series observations when performing time-based evaluation.
- Avoid using future information to construct historical predictions.
- Fit preprocessing transformations using training data only where applicable.
- Report evaluation metrics calculated from actual experiments.
- Distinguish public benchmark results from real-time sensor results.
- Document missing values, class imbalance, sampling frequency, and dataset limitations.

The availability of a dataset does not establish that it has already been downloaded, processed, or used in the experiments.

---

## 9. Model Evaluation and Performance Metrics

Each research component requires evaluation methods appropriate to its specific objective.

### 9.1 Anomaly Detection — Component 1

- **Precision:** Proportion of flagged observations that are actual anomalies.
- **Recall:** Proportion of actual anomalies detected.
- **F1-score:** Harmonic mean of precision and recall.
- **False-positive rate:** Proportion of normal observations incorrectly flagged.
- **Bandwidth reduction:** Reduction in transmitted data relative to the defined baseline.
- **Resource utilization:** Processing, memory, and other relevant edge-device costs.

### 9.2 Energy Forecasting — Component 2

- **MAE:** Average absolute difference between actual and predicted consumption.
- **RMSE:** Square root of the mean squared prediction error.
- **MAPE:** Average absolute percentage prediction error, with appropriate handling of zero or near-zero actual values.
- **R² score:** Measure of explained variance relative to a baseline.
- **Training time:** Time required to fit or update a model.
- **Inference time:** Time required to produce predictions.

Forecasting results should be reported separately for each relevant prediction horizon.

### 9.3 Appliance Scheduling — Component 3

- Accuracy, precision, recall, and F1-score for the defined scheduling evaluation task.
- Estimated energy savings against a clearly specified baseline.
- User comfort score using a documented evaluation method.
- Decision response time.
- Safety compliance under the defined test scenarios.

Energy savings and comfort results should be supported by measurements, simulation, or a clearly explained evaluation method rather than assumed from model accuracy.

### 9.4 Adaptive Learning — Component 4

- Prediction error before and after adaptation.
- MAE and RMSE under changing data conditions.
- Adaptation and model-update time.
- Resource and communication overhead.
- Performance under simulated or observed model drift.
- Evaluation of the federated learning procedure and raw-data handling.

### 9.5 Reporting Principles

All reported values must come from the corresponding experiment outputs. Baselines, test datasets, evaluation windows, preprocessing, and metric definitions should be documented so that results can be interpreted and reproduced.

---

## 10. Repository Structure

The following is a **recommended logical structure** for organizing the complete research repository. Adapt the directory names to the actual files in the repository rather than creating duplicate folders unnecessarily.

```text
J26-IT-389-G37/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── component1_edge_processing/
│   ├── data/
│   ├── notebooks/
│   ├── src/
│   ├── models/
│   ├── results/
│   └── README.md
│
├── component2_energy_prediction/
│   ├── data/
│   ├── notebooks/
│   ├── src/
│   ├── models/
│   ├── results/
│   └── README.md
│
├── component3_appliance_optimization/
│   ├── component3_dashboard.py
│   ├── data/
│   ├── models/
│   ├── v2_output/
│   ├── ablation_final_output/
│   ├── explainability_output/
│   ├── requirements.txt
│   └── README.md
│
├── component4_adaptive_intelligence/
│   ├── data/
│   ├── notebooks/
│   ├── src/
│   ├── models/
│   ├── results/
│   └── README.md
│
├── integration/
│   ├── mqtt/
│   ├── api/
│   └── device_control/
│
├── hardware/
│   ├── circuit_diagrams/
│   ├── firmware/
│   └── sensor_documentation/
│
├── docs/
│   ├── TAF/
│   ├── architecture/
│   ├── evaluation/
│   └── research_reports/
│
└── tests/
```

**Important:** This is an organizational proposal, not a verified listing of the current repository. Retain the actual filenames, model artifacts, experiment outputs, and existing Component 3 dashboard structure.

---

## 11. Installation and Setup

### 11.1 Prerequisites

The software-based components require a compatible Python environment. Hardware-related integration additionally requires the relevant microcontroller, sensors, and communication setup.

Recommended prerequisites include:

- Python compatible with the component's dependencies.
- pip.
- Git.
- A virtual environment.
- The required datasets and trained model artifacts.
- MQTT infrastructure for communication tests, where applicable.
- ESP32 hardware and sensor modules for physical experiments.

The exact Python version and dependency versions should be taken from the component-specific setup instructions.

### 11.2 Clone the Repository

Replace the placeholder with the actual repository URL.

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
```

### 11.3 Create a Virtual Environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 11.4 Install Dependencies

If a root-level `requirements.txt` exists and contains the dependencies needed for the selected workflow:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If dependencies are maintained separately for each component, activate the environment and install the requirements for the component you intend to run.

For example:

```bash
pip install -r component3_appliance_optimization/requirements.txt
```

Use the actual path present in the repository. Avoid combining incompatible dependency versions from different components into a single environment without testing them.

### 11.5 Prepare the Data

1. Obtain the required dataset from its documented source.
2. Place it in the configured data directory.
3. Review the dataset schema and expected columns.
4. Apply the documented preprocessing pipeline.
5. Confirm that the expected model artifacts and configuration files are available.
6. Follow the component-specific instructions before running training, evaluation, or dashboard commands.

Do not commit confidential data, credentials, API keys, or unnecessarily large generated files to the public repository.

---

## 12. Running the Prototype

### 12.1 Component 3 Dashboard

The known Component 3 dashboard entry point is `component3_dashboard.py`.

From the directory containing that file, activate the appropriate Python environment and run:

```bash
streamlit run component3_dashboard.py
```

Streamlit will display the local URL for the dashboard in the terminal.

Before running the dashboard:

- Install the correct Component 3 dependencies.
- Ensure that the required model artifacts are present.
- Check the configured data and output paths.
- Confirm that the saved pipeline is compatible with the installed Scikit-learn version.
- Review any component-specific configuration or setup documentation.

The documented Component 3 environment previously required `scikit-learn==1.3.2` for compatibility with a saved pipeline. Preserve the version specified in the actual working environment rather than upgrading it without validation.

### 12.2 Components 1, 2, and 4

Run each component using its own documented entry point, notebook, training script, or evaluation script.

The exact commands depend on the files present in the corresponding component directories. This README does not assume that a particular script, API, or dashboard exists for those components.

### 12.3 Hardware and MQTT Integration

Physical integration requires the appropriate firmware, sensor wiring, MQTT configuration, and device-control logic.

Before enabling appliance control:

- Verify the sensor readings.
- Test communication independently.
- Validate control commands.
- Verify the safety rules.
- Test failure and disconnection scenarios.
- Confirm that critical appliances cannot be switched off by an unsafe automated decision.

Physical hardware control should only be described as operational after successful integration and testing.

---

## 13. Expected System Capabilities

The following capabilities represent the intended overall research outcome.

| Capability | Description |
|---|---|
| Real-time energy monitoring | Collect and analyze household electricity measurements |
| Environmental monitoring | Incorporate temperature, humidity, occupancy, and light-related information |
| Edge preprocessing | Improve data quality before downstream analysis |
| Anomaly detection | Identify potentially abnormal readings |
| Multi-horizon forecasting | Predict energy consumption over different time horizons |
| Context-aware scheduling | Consider occupancy, user preferences, environmental conditions, and tariffs |
| Safety-aware decisions | Protect critical appliances through rule-based constraints |
| Appliance control | Support permitted switching and scheduling through compatible device interfaces |
| Adaptive intelligence | Investigate model updates in response to changing data |
| Privacy-preserving learning | Explore federated learning without routinely centralizing raw household data |

These are system-level objectives. Individual capabilities may remain at the research, implementation, testing, or integration stage.

---

## 14. Research Novelty and Contributions

The research proposes several contributions to intelligent household energy management.

### 14.1 Intelligent Edge Data Processing

Instead of transmitting all raw sensor readings, the proposed framework performs preprocessing and anomaly detection near the data source.

### 14.2 Hybrid Multi-Horizon Forecasting

The research combines EMA, ARIMA, and LSTM techniques within a horizon-based prediction framework.

### 14.3 Personalized Appliance Scheduling

The proposed decision engine considers predicted energy demand, occupancy, environmental conditions, electricity tariffs, and user preferences rather than relying exclusively on energy measurements.

### 14.4 Rule-Based Safety Layer

A dedicated safety layer is proposed to validate automated decisions before controlling critical appliances.

### 14.5 Adaptive and Privacy-Preserving Learning

The research investigates GRU-based adaptive learning, model drift detection, online learning, and federated learning to support adaptation while limiting raw household data transmission.

### 14.6 Unified Smart Home Energy Management

The four components are intended to work together within a unified workflow connecting sensing, preprocessing, forecasting, decision-making, device control, monitoring, and adaptation.

These contributions describe the proposed research design. Their effectiveness and novelty must ultimately be supported by implementation evidence, evaluation results, and comparison with relevant existing approaches.

---

## 15. Current Implementation Status

The project contains four related research components, but the completion of an individual component should not be interpreted as proof that the complete system has been integrated.

### Component 1 — Edge Data Processing

**Research scope:** Sensor data preprocessing, anomaly detection, and edge efficiency evaluation.

Implementation and hardware-integration status should be verified against the latest component source code, tests, and experiment reports.

### Component 2 — Energy Prediction

**Research scope:** Hybrid multi-horizon forecasting using EMA, ARIMA, and LSTM.

Model training, comparison, and forecasting results should be documented using the actual evaluation outputs.

### Component 3 — Appliance Scheduling and Optimization

**Known implementation artifacts:** A Streamlit dashboard, saved model and experiment artifacts, and evaluation-related outputs.

The previously documented Component 3 results include:

- Model accuracy: **84.7%**
- Macro F1-score: **84.3%**
- Safety compliance after validation: **100%**

These values are component-level results from the existing implementation context, not overall system performance. Their evaluation dataset, test conditions, baseline, and calculation procedure should be checked against the corresponding experiment reports before being used in a formal publication.

### Component 4 — Adaptive Edge Intelligence

**Research scope:** GRU-based adaptation, model drift detection, online learning, and federated learning.

The completion of each capability should be confirmed through its implementation artifacts and evaluation evidence.

### Overall Integration

The intended end-to-end system requires successful integration of sensor collection, edge preprocessing, MQTT communication, forecasting, scheduling, safety validation, appliance control, and adaptive learning.

Do not mark the complete system as fully operational until the relevant interfaces and end-to-end tests demonstrate that these parts work together.

---

## 16. Limitations and Future Improvements

The following are areas to evaluate and address during development and validation.

### 16.1 Dataset Limitations

Public energy datasets may differ from the target household environment in sampling frequency, sensor availability, appliance types, and consumption patterns.

Future evaluation should investigate whether models generalize to independently collected sensor data.

### 16.2 Forecasting Limitations

Forecasting performance may vary with the prediction horizon, data quality, seasonal changes, and unusual consumption patterns.

Future work should assess model performance separately across relevant horizons and under changing conditions.

### 16.3 Sensor Reliability

Missing readings, measurement noise, faulty sensors, and communication interruptions can affect prediction and decision-making.

The edge-processing component should be evaluated under these failure conditions.

### 16.4 Appliance Safety

Automated appliance control requires robust safeguards, particularly for critical devices.

The safety layer must be validated independently of the prediction model, and uncertain or unsafe decisions should not bypass control constraints.

### 16.5 Edge Resource Constraints

Machine learning and deep learning models may have different computational requirements on an ESP32 or other edge hardware.

Resource usage, model size, inference latency, and the allocation of computation between edge devices and servers require measurement.

### 16.6 Adaptive Learning and Privacy

Online and federated learning introduce challenges involving model stability, communication overhead, non-identical data distributions, and update validation.

Further evaluation should determine whether adaptation improves prediction performance without introducing unacceptable computational or privacy risks.

### 16.7 System Integration

The four components may have different software dependencies, data formats, and execution environments.

A unified integration layer should establish consistent message schemas, interfaces, error handling, and reproducible end-to-end tests.

---

## 17. Research Team

The following responsibilities are based on the member assignments in the Topic Assessment Form.

| Member | Registration Number | Research Responsibility |
|---|---|---|
| D. U. Dissanayaka | IT23265660 | Component 1 — Edge data processing and anomaly detection |
| T. G. N. C. Himasara | IT23284548 | Component 2 — Hybrid energy prediction |
| H. G. Fernando | IT23310582 | Component 3 — Appliance scheduling and energy optimization |
| M. D. Sarah | IT23219052 | Component 4 — Adaptive edge intelligence |

**Group:** G37  
**Research Project:** J26-IT-389  
**Supervisor named in the TAF:** Mr. Dhammika H. De Silva

---

## 18. References

The following references are listed in the Topic Assessment Form and provide background for the research.

1. H. Ghayvat, S. Mukhopadhyay, and X. Gui, “IoT-based smart energy management systems: A review,” *IEEE Access*, vol. 7, pp. 170–185, 2019.

2. J. Yang, Y. Xu, and M. Chen, “Smart meter data analytics for energy consumption prediction using machine learning,” *Energy and Buildings*, vol. 224, 2020.

3. A. Zanella, N. Bui, A. Castellani, et al., “Internet of Things for smart cities,” *IEEE Internet of Things Journal*, vol. 1, no. 1, pp. 22–32, 2014.

4. S. K. Singh and R. Singh, “Edge computing in IoT: A survey on architecture and applications,” *Journal of Systems Architecture*, vol. 97, pp. 238–254, 2019.

5. H. B. Jang, K. Y. Lee, and S. H. Kim, “Short-term load forecasting using LSTM and ARIMA hybrid model,” *IEEE Transactions on Smart Grid*, vol. 10, no. 4, pp. 4080–4090, 2019.

6. M. Usama, J. Qadir, and A. Al-Fuqaha, “Machine learning for energy consumption prediction in smart homes,” *IEEE Communications Surveys & Tutorials*, vol. 23, no. 2, pp. 1234–1256, 2021.

*Bibliographic details are reproduced from the TAF and should be verified against the original publications before formal academic submission.*

---

## License

No software license has been specified in this README. Add an appropriate license only after the project owners decide how the source code may be used, modified, and redistributed.

---

## Acknowledgment

This research is conducted as part of the IT4010 Research Project at the Sri Lanka Institute of Information Technology (SLIIT), under the Centre of Excellence for AI (CoEAI).

The project investigates the integration of IoT, edge computing, machine learning, and intelligent decision-making to improve household energy monitoring and management.
