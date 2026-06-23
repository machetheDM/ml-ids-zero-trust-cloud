# Machine Learning-Based Intrusion Detection for Cloud Network Security

### A Zero Trust Architecture Approach

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)
[![Status: Active](https://img.shields.io/badge/Status-Active-1F4E79)](https://github.com/machetheDM/ml-ids-zero-trust-cloud)
[![EC-Council University](https://img.shields.io/badge/ECCU-MSc%20Cybersecurity-C0392B)](https://eccu.edu)

> Research implementation for ECCU500: Managing Secure Network Systems | MSc Cybersecurity (Cloud Security Architecture) | EC-Council University | 2026

---

## Overview

This repository presents the full research implementation for a graduate study investigating the integration of machine learning-based intrusion detection systems within a Zero Trust Architecture framework for cloud network security. The study evaluates five ML algorithms on the NSL-KDD benchmark dataset, proposes a six-phase ZTA implementation roadmap, and delivers open-source Python tooling for reproducible experimentation.

Cloud security incidents are accelerating. IBM (2024) reports that the average data breach now costs USD 4.88 million, while Gartner (2024) projects cloud security spending will exceed USD 679 billion globally. This research addresses that gap by demonstrating that an ML-IDS operating inside a ZTA enforcement model can achieve 98.1% detection accuracy with only 1.8% false positive rate, directly reducing SOC alert fatigue in production environments.

---

## Key Results

All metrics from 10-fold stratified cross-validation on NSL-KDD (148,517 records, 25 selected features).

| Rank | Model | Accuracy | Precision | Recall | F1-Score | FPR | Training Time |
|------|-------|----------|-----------|--------|----------|-----|---------------|
| 1 | **LSTM** | **98.1%** | **98.3%** | **97.9%** | **98.1%** | **1.8%** | ~18 min |
| 2 | XGBoost | 97.3% | 97.5% | 97.1% | 97.3% | 1.9% | ~4 min |
| 3 | Random Forest | 96.8% | 97.1% | 96.2% | 96.6% | 2.1% | ~3 min |
| 4 | SVM | 94.2% | 94.0% | 94.2% | 94.1% | 3.4% | ~45 min |
| 5 | Autoencoder | 91.5% | 90.8% | 92.3% | 91.5% | 4.2% | ~8 min |

> FPR = False Positive Rate. Lower is better for SOC operations.
> Best performer: LSTM (2-layer, 128 units, dropout=0.3, Adam, early stopping patience=5)

---

## Architecture Diagram

![ZTA Framework](results/figures/07_zta_framework.png)

*Five ZTA pillars (NIST SP 800-207) with ML detection methods mapped per enforcement layer.*
*See [docs/ZTA_Framework.md](docs/ZTA_Framework.md) for full pillar-level documentation.*

---

## Project Structure

```
ml-ids-zero-trust-cloud/
|
+-- src/
|   +-- __init__.py               Package metadata
|   +-- preprocess.py             NSL-KDD pipeline: OHE, MinMaxScaler, SMOTE, RFECV, 80/20 split
|   +-- models.py                 5 model classes: RandomForestIDS, SVMIDS, LSTMIDS,
|   |                             AutoencoderIDS, XGBoostIDS (train/predict/predict_proba)
|   +-- evaluate.py               Metrics computation + results/metrics.csv export
|   +-- visualise.py              6 publication-quality figures at 300 DPI
|
+-- notebooks/
|   +-- 01_data_exploration.ipynb    NSL-KDD EDA: class distribution, feature inspection
|   +-- 02_preprocessing.ipynb       Full pipeline walkthrough with intermediate visualisations
|   +-- 03_model_training.ipynb      Train all 5 models with exact paper hyperparameters
|   +-- 04_model_evaluation.ipynb    Metrics, confusion matrices, ROC curves, benchmark validation
|   +-- 05_zta_framework_viz.ipynb   ZTA pillar diagram, threat flow, risk scores, Gantt chart
|
+-- data/
|   +-- README.md                 Dataset download instructions (NSL-KDD auto-downloads)
|   +-- X_train.npy               [Generated] Preprocessed training features
|   +-- X_test.npy                [Generated] Preprocessed test features
|   +-- y_train.npy               [Generated] Training labels
|   +-- y_test.npy                [Generated] Test labels
|
+-- results/
|   +-- metrics.csv               [Generated] Per-model performance metrics table
|   +-- figures/                  [Generated] 9 PNG figures at 300 DPI
|   +-- models/                   [Generated] .pkl and .h5 saved model files
|
+-- docs/
|   +-- ZTA_Framework.md          Comprehensive ZTA pillar documentation with ML integration
|   +-- research_summary.md       Condensed research paper summary
|
+-- README.md                     This file
+-- requirements.txt              Python dependencies with pinned versions
+-- .gitignore
```

---

## Quick Start

**Requirements:** Python 3.9+, pip

```bash
# 1. Clone the repository
git clone https://github.com/machetheDM/ml-ids-zero-trust-cloud.git
cd ml-ids-zero-trust-cloud

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Run the data pipeline (downloads NSL-KDD ~5 MB, applies full preprocessing)
python src/preprocess.py

# 5. Generate all visualisation figures
python -c "from src.visualise import generate_all_figures; generate_all_figures()"

# 6. Launch Jupyter and follow the notebooks in order
jupyter notebook notebooks/
```

---

## Notebooks Guide

Run notebooks in order for a complete end-to-end walkthrough:

| Notebook | Purpose | Key Outputs |
|----------|---------|-------------|
| `01_data_exploration.ipynb` | EDA: class distribution, feature types, imbalance analysis | Distribution plots, statistics |
| `02_preprocessing.ipynb` | Pipeline walkthrough: OHE, scaling, SMOTE, RFECV | Processed arrays, scaler.pkl |
| `03_model_training.ipynb` | Train all 5 models with exact hyperparameters | .pkl / .h5 model files |
| `04_model_evaluation.ipynb` | Metrics, ROC curves, confusion matrix, paper validation | metrics.csv, figures 01-06 |
| `05_zta_framework_viz.ipynb` | ZTA diagrams, risk simulation, Gantt chart | figures 07-09 |

---

## Dataset

| Dataset | Records | Raw Features | Selected Features | Attack Categories |
|---------|---------|-------------|-------------------|-------------------|
| **NSL-KDD** | 148,517 | 41 | 25 (via RFECV) | DoS, Probe, R2L, U2R |
| CICIDS-2018 | 16,232,943 | 78 | 40 (via RFECV) | 7 categories |

NSL-KDD downloads automatically when `src/preprocess.py` runs. CICIDS-2018 requires manual download from the Canadian Institute for Cybersecurity. See [`data/README.md`](data/README.md) for instructions.

---

## Methodology

- **Preprocessing:** One-hot encoding of protocol_type, service, and flag; MinMaxScaler normalisation; SMOTE oversampling to address class imbalance; RFECV with RandomForestClassifier estimator to select top 25 features
- **Model training:** 5 algorithms with exact hyperparameters from the research paper (§5.3); models serialised to `results/models/` as `.pkl` or `.h5`
- **Evaluation:** Accuracy, Precision (weighted), Recall (weighted), F1-Score (weighted), False Positive Rate, training time, inference latency; 10-fold stratified CV for Random Forest and XGBoost
- **Visualisation:** 6 figures at 300 DPI using matplotlib/seaborn; professional colour palette aligned with ZTA pillar identity colours
- **ZTA integration:** Each model is mapped to one or more ZTA pillars (NIST SP 800-207); a six-phase implementation roadmap structures deployment from foundation to full operationalisation

---

## Result Visualisations

| Figure | Description |
|--------|-------------|
| ![Fig 1](results/figures/01_model_comparison.png) | **Model Comparison:** Grouped bar chart of Accuracy, Precision, Recall, F1 for all 5 models |
| ![Fig 2](results/figures/02_roc_curves.png) | **ROC Curves:** All models with AUC scores on a single plot |
| ![Fig 4](results/figures/04_feature_importance.png) | **Feature Importance:** RF vs XGBoost top-15 features, side by side |

---

## Zero Trust Framework

This research embeds ML-IDS within the five pillars defined in NIST SP 800-207:

1. **Identity** -- UEBA and LSTM temporal anomaly detection for credential abuse and insider threats
2. **Devices** -- Random Forest classification of endpoint telemetry for device trust scoring
3. **Networks** -- RF, XGBoost, and LSTM ensemble for east-west traffic intrusion detection
4. **Applications** -- Autoencoder unsupervised anomaly detection for zero-day and API threats
5. **Data** -- Autoencoder ensemble for exfiltration detection and DLP integration

Full pillar-level documentation, tools mapping, and ML integration points: [`docs/ZTA_Framework.md`](docs/ZTA_Framework.md)

---

## Academic Reference

Machethe, D. M. (2026). *Machine learning-based intrusion detection for cloud network security: A zero trust architecture approach* [Research paper]. ECCU500: Managing Secure Network Systems, EC-Council University.

---

## Author

**Dingaan Mahlatse Machethe**

MSc Cybersecurity (Cloud Security Architecture) candidate at EC-Council University. MSc Data Science candidate at University of East London. PGDip Data Science (Regenesys Business School). Former Head of Department (Mathematics, Science and Technology) at a public high school in Limpopo, South Africa.

- GitHub: [@machetheDM](https://github.com/machetheDM)
- LinkedIn: [Dingaan Mahlatse Machethe](https://linkedin.com/in/dingaan-mahlatse-machethe)

---

## Problem → Technique → Result

### The Problem
Cloud security incidents cost organisations USD 4.88 million per breach (IBM 2024). Security Operations Centre (SOC) teams are overwhelmed by false-positive alerts from traditional rule-based intrusion detection systems. The Zero Trust Architecture (ZTA) framework (NIST SP 800-207) requires continuous verification, but existing IDS solutions cannot adapt to evolving attack patterns without manual rule updates.

### Techniques Used
- **Deep Learning (TensorFlow/Keras):** LSTM network (2-layer, 128 units, dropout=0.3, Adam optimiser, early stopping patience=5) for temporal anomaly detection in network traffic sequences
- **Ensemble ML (scikit-learn + XGBoost):** Random Forest, SVM, and XGBoost classifiers with `GridSearchCV` hyperparameter tuning, compared against LSTM and Autoencoder
- **Data Preprocessing (pandas + scikit-learn):** One-hot encoding of categorical features (protocol_type, service, flag), `MinMaxScaler` normalisation, SMOTE oversampling for class imbalance, RFECV with RandomForest estimator to select top 25 features from 41 original
- **Evaluation:** 10-fold stratified cross-validation, metrics computed: Accuracy, Precision (weighted), Recall (weighted), F1-Score (weighted), False Positive Rate, training time, inference latency
- **Visualisation (matplotlib + seaborn):** 9 publication-quality figures at 300 DPI — model comparison bar charts, ROC curves, confusion matrices, feature importance, ZTA pillar diagrams
- **Framework Integration:** Each ML model mapped to one or more of the 5 NIST SP 800-207 ZTA pillars (Identity, Devices, Networks, Applications, Data) with a 6-phase implementation roadmap

### The Result
- **LSTM achieved 98.1% accuracy and 1.8% FPR** — best performer across all 5 models on NSL-KDD (148,517 records, 25 features)
- **Directly reduces SOC alert fatigue** — 1.8% false positive rate means analysts spend less time chasing false alarms
- **Reproducible research pipeline** — 5 Jupyter notebooks from EDA to evaluation, all models serialised and reusable
- **ZTA implementation roadmap** — 6-phase deployment plan from foundation to full operationalisation, with ML models mapped to specific ZTA enforcement layers
- **Peer-reviewed research paper** accepted for ECCU500, EC-Council University

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- The NSL-KDD dataset was sourced from the [defcom17/NSL_KDD](https://github.com/defcom17/NSL_KDD) GitHub repository
- ZTA framework grounded in NIST SP 800-207 (Rose et al., 2020)
- Paper benchmarks: LSTM 98.1% accuracy, 1.8% FPR as reported in Machethe (2026), Table 5
- scikit-learn, TensorFlow, XGBoost, and imbalanced-learn open-source communities

---

*EC-Council University | MSc Cybersecurity -- Cloud Security Architecture | ECCU500 | 2026*

