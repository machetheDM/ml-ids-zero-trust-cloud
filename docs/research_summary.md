# Research Summary

## Machine Learning-Based Intrusion Detection for Cloud Network Security: A Zero Trust Architecture Approach

**Author:** Dingaan Mahlatse Machethe  
**Institution:** EC-Council University, Albuquerque, New Mexico  
**Programme:** MSc Cybersecurity — Cloud Security Architecture Concentration  
**Module:** ECCU500: Managing Secure Network Systems  
**Submission Date:** June 4, 2026

---

## Abstract

The convergence of cloud computing and the increasingly sophisticated threat landscape has rendered traditional perimeter-based network security models fundamentally insufficient. This research investigates the integration of ML-based intrusion detection systems (IDS) within a Zero Trust Architecture (ZTA) framework as a comprehensive security strategy for cloud network environments.

The study systematically evaluates five machine learning algorithms — **Random Forest, SVM, LSTM, Autoencoder, and XGBoost** — against the **NSL-KDD and CICIDS-2018** benchmark datasets. Experimental results demonstrate that an ensemble approach combining LSTM temporal analysis with Random Forest classification achieves a detection accuracy of **98.1%**, a precision of **98.3%**, and a false positive rate of **1.8%**.

The research proposes a **six-phase implementation framework** that embeds ML-IDS capabilities within each of the **five ZTA pillars**, providing cloud security architects with a structured, phased roadmap for deploying both capabilities in concert.

**Keywords:** machine learning, intrusion detection system, Zero Trust Architecture, cloud security, network anomaly detection, deep learning, cyber threat intelligence, NIST SP 800-207

---

## Research Problem

- Cloud environments have eliminated the traditional security perimeter — east-west traffic is invisible to perimeter-based monitoring
- Signature-based IDS cannot detect zero-day exploits, novel malware, or living-off-the-land techniques
- 71% of cloud-targeted attacks in 2024 were fileless (CrowdStrike, 2024)
- Average breach dwell time: 280 days in cloud environments (IBM Security, 2024)
- 74% of SOC analysts overwhelmed by alert volumes (Panaseer, 2023)
- **Research gap:** No existing unified, operationalised framework integrating ML-IDS within all ZTA pillars for cloud-native deployments

---

## Methodology

### Datasets

| Dataset | Records | Features | Attack Categories |
|---------|---------|----------|-------------------|
| NSL-KDD | 148,517 | 41 → **25** (post RFECV) | DoS, Probe, R2L, U2R |
| CICIDS-2018 | 16,232,943 | 78 → **40** (post RFECV) | 7 categories |

### Preprocessing Pipeline

1. Download NSL-KDD (GitHub raw source)
2. Assign 42 standard column names
3. One-hot encode: `protocol_type`, `service`, `flag`
4. MinMaxScaler normalisation
5. SMOTE oversampling (class imbalance correction)
6. RFECV with RandomForestClassifier — top 25 features
7. Stratified 80/20 train-test split

### ML Algorithm Configurations

| Algorithm | Key Hyperparameters |
|-----------|---------------------|
| Random Forest | 200 trees, Gini, depth=unrestricted |
| SVM | RBF kernel, C=10, γ=0.001 |
| LSTM | 2-layer, 128 units, dropout=0.3, Adam, 50 epochs, patience=5, seq_len=20 |
| Autoencoder | 41→32→16→8 encoder, threshold=95th percentile |
| XGBoost | 500 estimators, lr=0.05, depth=6, subsample=0.8 |

---

## Results

### Model Performance (NSL-KDD, 10-fold stratified CV)

| Algorithm | Accuracy | Precision | Recall | F1-Score | FPR |
|-----------|----------|-----------|--------|----------|-----|
| **LSTM ⭐** | **98.1%** | **98.3%** | **97.9%** | **98.1%** | **1.8%** |
| XGBoost | 97.3% | 97.5% | 97.1% | 97.3% | 1.9% |
| Random Forest | 96.8% | 97.1% | 96.2% | 96.6% | 2.1% |
| SVM | 94.2% | — | — | — | 3.4% |
| Autoencoder | 91.5% | 90.8% | 92.3% | 91.5% | 4.2% |
| Isolation Forest | 89.3% | 88.5% | 90.1% | 89.3% | 5.1% |

### Key Findings

1. **LSTM achieves highest overall performance** — 98.1% accuracy, 1.8% FPR. Superiority most pronounced for Probe (+5.3pp) and R2L (+7.8pp) attack categories vs Random Forest, due to temporal sequence modelling of port enumeration and credential-stuffing patterns.

2. **Hybrid LSTM + Random Forest ensemble** is optimal for production deployment — LSTM handles sequential flow analysis; Random Forest handles high-volume classification with high interpretability.

3. **Unsupervised models (Autoencoder, Isolation Forest) excel at novel attack detection** — detected attack variants not represented in training data, at the cost of higher FPR.

4. **XGBoost best balance of performance and speed** — 97.3% accuracy with fast inference, suitable for high-volume east-west traffic.

5. **FPR is the critical operational metric** — LSTM's 1.8% FPR vs SVM's 3.4% FPR translates to ~60 fewer false alerts per 10,000 flows, directly reducing SOC alert fatigue.

---

## ZTA Framework

### 5 Pillars (NIST SP 800-207)

1. **Identity** — Continuous authentication, UEBA anomaly detection
2. **Devices** — Device trust scoring, endpoint behavioural baselines
3. **Networks** — ML-driven micro-segmentation, flow classification
4. **Applications** — API anomaly detection, application-layer threat intelligence
5. **Data** — Autoencoder exfiltration detection, DLP integration

### 6 Implementation Phases

| Phase | Focus | Timeline |
|-------|-------|----------|
| 1 — Foundation | ZTA policy, identity baseline | Weeks 1–4 |
| 2 — Data Collection | Network taps, SIEM, PCAP | Weeks 5–8 |
| 3 — Model Development | Training, cross-validation | Weeks 9–14 |
| 4 — Integration | SIEM deployment, auto-response | Weeks 15–18 |
| 5 — Validation | Red team, pen testing, tuning | Weeks 19–22 |
| 6 — Operationalisation | SOC playbooks, dashboards | Ongoing |

---

## Recommendations

1. Deploy **LSTM + Random Forest ensemble** as primary cloud IDS
2. Supplement with **Autoencoder** for zero-day and data exfiltration detection
3. Target **FPR ≤ 2%** in production deployments
4. Implement ML-IDS across **all five ZTA pillars** — not just at the network perimeter
5. Establish **active learning** pipeline for continuous model improvement
6. Conduct **quarterly red team exercises** to validate adversarial robustness

---

## Research Contribution

This research addresses an identified gap in the literature: the absence of a unified ML-ZTA integration framework suitable for cloud-native deployments. The six-phase operationalisation roadmap provides cloud security architects with actionable, vendor-agnostic guidance applicable to AWS, Azure, and GCP environments.

---

## Full References

See [`README.md`](../README.md#references) for the complete APA bibliography.

---

*Machethe, D.M. (2026). Machine Learning-Based Intrusion Detection for Cloud Network Security: A Zero Trust Architecture Approach. EC-Council University, ECCU500: Managing Secure Network Systems.*
