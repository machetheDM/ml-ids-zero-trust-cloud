# ZTA Integration Framework

**Machine Learning-Based Intrusion Detection for Cloud Network Security: A Zero Trust Architecture Approach**

*Source: Machethe, D.M. (2026). ECCU500: Managing Secure Network Systems. EC-Council University.*

---

## Overview

This document describes the six-phase ML-IDS implementation framework embedded within the five pillars of the Zero Trust Architecture (ZTA), as proposed in the research paper (§5.5, Table 4).

The framework is grounded in **NIST Special Publication 800-207** (Rose et al., 2020) and operationalises the principle of *never trust, always verify* through ML-driven continuous monitoring at every ZTA enforcement layer.

---

## Zero Trust Architecture — 5 Pillars (NIST SP 800-207)

| Pillar | Description | Primary ML Detection Method |
|--------|-------------|----------------------------|
| **1. Identity** | Continuous authentication, MFA, UEBA | UEBA clustering, LSTM anomaly detection |
| **2. Devices** | Device trust scoring, endpoint health | Behaviour baselines, Random Forest classification |
| **3. Networks** | Micro-segmentation, east-west traffic analysis | Random Forest, SVM, LSTM for flow classification |
| **4. Applications** | API security, application-layer threat intelligence | Autoencoder anomaly detection, XGBoost |
| **5. Data** | Data classification, exfiltration prevention | Autoencoder, Isolation Forest |

---

## Six-Phase Implementation Framework

### Phase 1: Foundation (Weeks 1–4)

**Focus:** ZTA policy design and baseline establishment

**Key Activities:**
- Design ZTA policy framework — define trust levels, access tiers, microsegment boundaries
- Establish identity baseline — inventory all users, service accounts, and workload identities
- Conduct network inventory — map all assets, data flows, and east-west traffic patterns
- Define ML-IDS requirements — specify detection targets per attack category

**Tools:** Azure AD, Okta, AWS IAM, network discovery scanners

**ZTA Pillars Addressed:** Identity, Devices

---

### Phase 2: Data Collection (Weeks 5–8)

**Focus:** Instrumentation and telemetry pipeline

**Key Activities:**
- Deploy network taps and packet capture agents at micro-segment boundaries
- Configure SIEM to ingest NetFlow, Zeek logs, and endpoint telemetry
- Collect PCAP samples — baseline normal traffic profiles for Autoencoder training
- Validate data quality — ensure coverage of NSL-KDD equivalent attack categories

**Tools:** Zeek (formerly Bro), Suricata, Splunk, ELK Stack

**ZTA Pillars Addressed:** Networks, Applications

---

### Phase 3: Model Development (Weeks 9–14)

**Focus:** Feature engineering and model training

**Key Activities:**
- Apply preprocessing pipeline (`src/preprocess.py`) — OHE, MinMaxScaler, SMOTE, RFECV
- Train all five models with exact hyperparameters from §5.3:
  - Random Forest: 200 trees, Gini, unrestricted depth
  - SVM: RBF, C=10, gamma=0.001
  - LSTM: 2-layer, 128 units, dropout=0.3, Adam, 50 epochs, patience=5
  - Autoencoder: 41→32→16→8 encoder, 95th-percentile threshold
  - XGBoost: 500 estimators, lr=0.05, depth=6, subsample=0.8
- Validate against paper benchmarks (LSTM target: 98.1% accuracy, 1.8% FPR)
- Conduct 10-fold stratified cross-validation

**Tools:** Python, scikit-learn, TensorFlow/Keras, XGBoost

**ZTA Pillars Addressed:** All five pillars (model per use-case mapping)

---

### Phase 4: Integration (Weeks 15–18)

**Focus:** Deploy ML engine into production ZTA enforcement stack

**Key Activities:**
- Deploy LSTM + Random Forest ensemble as the primary detection engine
- Integrate ML output into SIEM — map detection scores to ZTA policy decisions
- Configure automated response playbooks — dynamically adjust access policy on alert
- Deploy Autoencoder for real-time data exfiltration monitoring on the Data pillar
- Implement feedback loop — detection results trigger active learning retraining queue

**Tools:** AWS GuardDuty, Microsoft Sentinel, Azure Defender for Cloud

**ZTA Pillars Addressed:** Networks, Applications, Data

---

### Phase 5: Validation (Weeks 19–22)

**Focus:** Red team exercises and model hardening

**Key Activities:**
- Conduct controlled red team exercises covering all 4 NSL-KDD attack categories
- Execute penetration tests targeting each ZTA pillar
- Measure FPR in production — target ≤1.8% (LSTM benchmark from paper §6.2)
- Tune decision thresholds — minimise SOC alert fatigue (Panaseer, 2023: 74% SOC overwhelm)
- Test adversarial robustness — feature perturbation attacks against ML models (Apruzzese et al., 2022)

**Tools:** Metasploit, OWASP ZAP, Kali Linux

**ZTA Pillars Addressed:** All five pillars

---

### Phase 6: Operationalisation (Ongoing)

**Focus:** Continuous monitoring and SOC integration

**Key Activities:**
- Publish SOC runbooks — one playbook per attack category (DoS, Probe, R2L, U2R)
- Deploy real-time dashboards — accuracy, FPR, detection volume trends
- Schedule retraining cadence — monthly model refresh against new traffic baselines
- Implement drift detection — alert when model accuracy degrades >2 percentage points
- Report to CISO — monthly metrics, benchmark validation against paper Table 5 targets

**Tools:** Grafana, PagerDuty, SOAR platform

**ZTA Pillars Addressed:** All five pillars

---

## Threat Taxonomy and ML-ZTA Detection Mapping

Based on paper Table 2:

| Threat Category | Attack Examples | ML Detection Method | Zero Trust Layer | Severity |
|----------------|----------------|---------------------|-----------------|----------|
| Network Intrusion | Port scanning, SYN floods, MITM | Random Forest, SVM | Network Micro-seg. | HIGH |
| Lateral Movement | Pass-the-hash, Kerberoasting, RDP abuse | LSTM, Graph Analysis | Identity & Access | CRITICAL |
| Data Exfiltration | DNS tunnelling, covert channels | Autoencoder, Isolation Forest | Data Security | CRITICAL |
| Insider Threat | Privilege abuse, abnormal access patterns | UEBA, Clustering | Identity & Access | HIGH |
| Zero-Day Exploit | Unknown CVEs, novel malware | Autoencoder, GAN-based | Application Layer | CRITICAL |
| DDoS / DoS | Volumetric, reflection, application layer | XGBoost, LSTM | Network Micro-seg. | HIGH |
| Misconfiguration | Open S3 buckets, overpermissioned IAM | Policy analysis, CSPM | Device Trust | MEDIUM |

---

## Key Recommendations (Paper §7)

1. **Deploy LSTM + Random Forest ensemble** as the primary production IDS — 98.1% accuracy, 1.8% FPR
2. **Use Autoencoder for zero-day detection** — unsupervised approach detects novel attacks invisible to supervised models
3. **Target FPR ≤ 2%** in production — above this threshold, SOC alert fatigue degrades analyst effectiveness
4. **Embed ML-IDS in all five ZTA pillars** — perimeter-only deployment leaves east-west traffic unmonitored
5. **Implement active learning** — continuous retraining on confirmed incidents prevents model drift
6. **Conduct quarterly red team exercises** — validate model robustness against adversarial perturbations

---

## References

- Rose, S., Borchert, O., Mitchell, S., & Connelly, S. (2020). *Zero trust architecture (NIST Special Publication 800-207)*. NIST. https://doi.org/10.6028/NIST.SP.800-207
- Apruzzese, G., Colajanni, M., Ferretti, L., & Marchetti, M. (2022). Addressing adversarial attacks against security systems based on machine learning. *Computers and Security, 105*, 102231.
- Panaseer. (2023). *Security leaders peer report 2023*. Panaseer Ltd.
- Verizon. (2024). *2024 data breach investigations report*. Verizon Communications Inc.

---

*Document generated from research paper: Machethe, D.M. (2026). Machine Learning-Based Intrusion Detection for Cloud Network Security: A Zero Trust Architecture Approach. EC-Council University, ECCU500.*
