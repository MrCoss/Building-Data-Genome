# Building Energy Anomaly Detection using BDG2 (End-to-End ML Pipeline)

## 1. Executive Summary

This project implements a **production-style, end-to-end Machine Learning pipeline** to detect anomalous building energy behavior using the **Building Data Genome Project 2 (BDG2)** dataset. The solution supports operational teams by identifying unusual consumption patterns (spikes, drops, irregular usage) that may indicate equipment issues, scheduling problems, meter faults, or inefficient control strategies.

The pipeline is designed for **large-scale time-series data** and executed in **Google Colab** with a **memory-safe batch processing workflow**, ensuring stability even when processing multi-meter datasets across multiple buildings and sites.

**Key Deliverables Produced by This Project**

* Automated dataset download and validation
* Memory-safe preprocessing (wide → long conversion using batch Parquet storage)
* Weather + building metadata enrichment
* Feature engineering tailored for anomaly detection
* Unsupervised multi-model anomaly detection
* Ensemble voting and prediction outputs
* Saved models, plots, logs, and business insights report
* Full project packaged as a downloadable ZIP

---

## 2. Business Context and Problem Statement

### 2.1 Why anomaly detection matters

Building energy consumption is influenced by many factors:

* occupancy patterns
* HVAC schedules and setpoints
* seasonal weather variations
* operational controls and maintenance

In real facility operations, abnormal energy behavior can result in:

* energy waste and increased utility cost
* undetected equipment faults
* degraded occupant comfort
* higher maintenance expense due to delayed action

### 2.2 Problem statement

Manual monitoring does not scale for:

* multiple buildings
* multiple meters (electricity, chilled water, steam, gas, etc.)
* long time periods

**Objective:** Detect anomalous energy readings from time-series meter data using an automated ML pipeline and provide business-ready insights for operational decision making.

---

## 3. Dataset Overview

### 3.1 Dataset

**Building Data Genome Project 2 (BDG2)**

* Hourly meter data covering **2016 and 2017**
* Multiple meters per building
* Includes building metadata and weather observations

### 3.2 Files used

The project uses the following BDG2 files:

**Raw meters (wide format)**

* `electricity.csv`
* `chilledwater.csv`
* `steam.csv`
* `hotwater.csv`
* `gas.csv`
* `water.csv`
* `irrigation.csv`
* `solar.csv`

**Contextual data**

* `metadata.csv` (building/site information)
* `weather.csv` (hourly weather per site)

---

## 4. Project Objectives

### Technical objectives

* Build a stable preprocessing pipeline for large time-series datasets
* Engineer meaningful features for anomaly detection
* Train multiple unsupervised models and combine results using an ensemble
* Save models and outputs for reproducibility and deployment readiness

### Business objectives

* Identify when and where abnormal energy patterns occur
* Provide actionable building-level insights
* Support continuous monitoring workflows

---

## 5. Technology Stack

### Languages

* Python

### Environment

* Google Colab

### Libraries

* Data Processing: `pandas`, `numpy`
* ML Models: `scikit-learn`
* Visualization: `matplotlib`, `seaborn`
* Model Persistence: `joblib`
* Performance / Scaling: `pyarrow` (Parquet)

---

## 6. Solution Design (Pipeline Architecture)

The solution is implemented as a structured notebook with clearly separated stages.

### 6.1 Stage A — Project Setup

A standardized enterprise folder structure is created:

* `data/raw/` : original downloaded files
* `data/processed/` : final datasets and intermediate outputs
* `data/processed/long_batches/` : Parquet batch outputs
* `models/` : trained model artifacts
* `outputs/` : CSV outputs (predictions, summaries)
* `plots/` : all EDA and evaluation images
* `reports/` : executive insights, summaries
* `logs/` : pipeline logs

---

### 6.2 Stage B — Automated Dataset Download

To avoid manual dataset handling, the pipeline uses `kagglehub` for automated download and extraction.

Key implementation features:

* automatic download into Colab cache
* copy to project raw folder for reproducibility
* validation for expected 18 files

---

### 6.3 Stage C — Data Validation and Quality Checks

Before transformation, the pipeline validates:

* file availability and sizes
* column presence (especially `timestamp`)
* schema preview for meter, weather, and metadata

This ensures the dataset is usable and prevents downstream failures.

---

### 6.4 Stage D — Memory-Safe Preprocessing (Wide → Long)

BDG2 meters are stored in **wide format** with:

* `timestamp`
* hundreds/thousands of building columns

For modeling, the pipeline converts data into **long format**:

```
timestamp | building_id | meter_type | value
```

Because long-format conversion can exceed Colab memory limits, the project implements **batch conversion**:

* process building columns in chunks
* melt only a subset of columns at a time
* save results to Parquet files in `data/processed/long_batches/`

This design ensures stability and scalability.

---

### 6.5 Stage E — Merging Context (Weather + Metadata)

Each batch is enriched with:

**Building metadata**

* `site_id`
* `primaryspaceusage`
* `sqm`, `sqft`
* `timezone`

**Weather features**

* `airTemperature`
* `dewTemperature`
* `windSpeed`
* `windDirection`
* `cloudCoverage`
* `precipDepth1HR`
* `seaLvlPressure`

Merging is performed via:

* `(building_id → metadata)`
* `(site_id + timestamp → weather)`

---

### 6.6 Stage F — Final Dataset Construction (Streaming Write)

To create a stable modeling dataset, the pipeline builds the final CSV using **streaming writes**:

* read parquet batch
* sample rows to keep dataset manageable
* append to a single CSV file

This ensures:

* multi-meter coverage
* stable dataset size
* Colab-friendly workflow

Final dataset saved as:

* `data/processed/final_preprocessed_dataset.csv`

---

## 7. Exploratory Data Analysis (EDA)

The project includes EDA on both raw data and merged/preprocessed data.

### EDA outputs include:

* meter type distribution
* energy value distributions (raw + log scale)
* missing value checks
* building-level usage ranking
* daily consumption trends
* weather relationship analysis

All plots are exported to:

* `plots/`

---

## 8. Feature Engineering

Feature engineering is designed specifically for time-series anomaly detection:

### 8.1 Rolling statistics

* rolling mean (7 days / 168 hours)
* rolling standard deviation (7 days / 168 hours)

### 8.2 Deviation score

A normalized deviation score:

```
deviation = (value - rolling_mean) / (rolling_std + ε)
```

### 8.3 Lag features

* lag-1 (previous hour)
* lag-24 (previous day same hour)

### 8.4 Time features

* hour
* day of week
* month
* weekend flag

---

## 9. Model Training and Detection Strategy

### 9.1 Modeling approach

Anomaly detection is performed using **unsupervised learning**, since anomaly labels are typically unavailable in operational settings.

### 9.2 Models implemented

* **Isolation Forest** (tree-based outlier detection)
* **Local Outlier Factor (LOF)** with novelty mode
* **Robust Covariance (EllipticEnvelope)**

### 9.3 Scaling strategy

The project uses **RobustScaler** to reduce sensitivity to extreme values.

### 9.4 Ensemble voting

Each model outputs `normal` or `anomaly`. The pipeline assigns a final anomaly label using:

* anomaly votes from each model
* anomaly = 1 if at least 2/3 models agree

This reduces false positives and improves reliability.

---

## 10. Outputs and Artifacts

### 10.1 Saved models

Stored in `models/`:

* `scaler.pkl`
* `isolation_forest.pkl`
* `lof_model.pkl`
* `robust_cov.pkl`
* `feature_list.pkl`

### 10.2 Prediction outputs

Stored in `outputs/`:

* `anomaly_predictions.csv`

### 10.3 Visualizations

Stored in `plots/`:

* EDA plots
* model evaluation plots
* anomaly distribution and time-series plots

### 10.4 Reports

Stored in `reports/`:

* `business_insights_summary.txt`

---

## 11. Business Insights and Operational Value

This pipeline supports building operations by enabling:

* detection of abnormal usage patterns
* prioritization of high-risk buildings
* peak hour identification for targeted monitoring
* understanding of weather-driven consumption behavior

Typical operational outcomes:

* reduced energy waste
* faster issue investigation
* improved equipment reliability
* better decision making using data-backed evidence

---

## 12. How to Run (Recommended Execution Order)

1. Run project setup and install cells
2. Download BDG2 automatically
3. Validate raw files
4. Run batch preprocessing to Parquet
5. Build merged dataset in streaming mode
6. Run EDA
7. Run feature engineering
8. Train models
9. Save artifacts and export predictions
10. Generate business insights report
11. Download final ZIP package

---

## 13. Repository Structure

```text
bdg2_energy_anomaly_detection/
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── long_batches/
│       └── final_preprocessed_dataset.csv
│
├── models/
│   ├── scaler.pkl
│   ├── isolation_forest.pkl
│   ├── lof_model.pkl
│   ├── robust_cov.pkl
│   └── feature_list.pkl
│
├── outputs/
│   └── anomaly_predictions.csv
│
├── plots/
│   └── *.png
│
├── reports/
│   └── business_insights_summary.txt
│
├── logs/
│   └── pipeline.log
│
├── config.json
└── README.md
```

---

## 14. Interview Talking Points

This project demonstrates practical capability in:

* large-scale time-series preprocessing under hardware constraints
* batch ETL strategy using Parquet intermediates
* feature engineering for real operational anomaly detection
* robust unsupervised ML modeling and ensemble voting
* production-ready outputs including models, logs, plots, and reports

---

## 15. Future Enhancements

* Train specialized models per meter type
* Implement automatic threshold tuning and sensitivity controls
* Build a Streamlit dashboard for real-time monitoring
* Add periodic retraining and drift monitoring
* Integrate alerting workflows (email/webhooks)

---

## 16. License and Usage

This repository is intended for educational and portfolio demonstration. Dataset usage should follow the licensing terms provided by the BDG2 dataset source.
