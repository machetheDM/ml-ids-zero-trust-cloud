# ML-IDS Zero Trust Cloud

**Machine Learning-Based Intrusion Detection for Cloud Network Security: A Zero Trust Architecture Approach**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![EC-Council University](https://img.shields.io/badge/ECCU-MSc%20Cybersecurity-red)](https://eccu.edu)

---

## Overview

This repository contains the implementation and documentation for a research project submitted to **EC-Council University** in partial fulfilment of **ECCU500: Managing Secure Network Systems** (MSc Cybersecurity — Cloud Security Architecture concentration).

The research investigates the integration of ML-based Intrusion Detection Systems (IDS) within a **Zero Trust Architecture (ZTA)** framework as a comprehensive security strategy for cloud network environments. Five machine learning algorithms are evaluated against two industry-standard benchmark datasets, and a six-phase implementation roadmap is proposed for cloud security architects.

**Author:** Dingaan Mahlatse Machethe  
**Institution:** EC-Council University, Albuquerque, New Mexico  
**Module:** ECCU500 — Managing Secure Network Systems  
**Submission Date:** June 4, 2026

---

## Research Findings Summary

| Model | Accuracy | Precision | Recall | F1-Score | FPR |
|-------|----------|-----------|--------|----------|-----|
| **LSTM** ⭐ | **98.1%** | **98.3%** | **97.9%** | **98.1%** | **1.8%** |
| XGBoost | 97.3% | 97.5% | 97.1% | 97.3% | 1.9% |
| Random Forest | 96.8% | 97.1% | 96.2% | 96.6% | 2.1% |
| SVM | 94.2% | — | — | — | 3.4% |
| Autoencoder | 91.5% | 90.8% | 92.3% | 91.5% | 4.2% |
| Isolation Forest | 89.3% | 88.5% | 90.1% | 89.3% | 5.1% |

> All metrics averaged across 10-fold stratified cross-validation on NSL-KDD dataset.  
> **Best performer:** Hybrid LSTM + Random Forest ensemble — **98.1% accuracy, 1.8% FPR**

---

## Repository Structure

```
ml-ids-zero-trust-cloud/
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── .gitignore
├── notebooks/
│   ├── 01_data_exploration.ipynb     # NSL-KDD & CICIDS-2018 EDA
│   ├── 02_preprocessing.ipynb        # Feature engineering pipeline
│   ├── 03_model_training.ipynb       # All 5 algorithm training
│   ├── 04_model_evaluation.ipynb     # Performance comparison
│   └── 05_zta_framework_viz.ipynb    # ZTA integration diagrams
├── src/
│   ├── __init__.py
│   ├── preprocess.py                 # Data pipeline (NSL-KDD)
│   ├── models.py                     # ML model definitions & training
│   ├── evaluate.py                   # Metrics & benchmarking
│   └── visualise.py                  # Plotting & ZTA diagrams
├── data/
│   └── README.md                     # Dataset instructions
├── results/
│   ├── figures/                      # Generated plots
│   └── models/                       # Saved .pkl / .h5 model files
└── docs/
    ├── ZTA_Framework.md              # ZTA integration framework
    └── research_summary.md           # Full research summary
```

---

## Five ML Algorithms — Exact Configurations (from paper)

### 1. Random Forest
- **Trees:** 200 decision trees
- **Criterion:** Gini impurity
- **Max depth:** Unrestricted
- **Library:** scikit-learn

### 2. Support Vector Machine (SVM)
- **Kernel:** Radial Basis Function (RBF)
- **C:** 10
- **Gamma:** 0.001
- **Library:** scikit-learn SVC

### 3. Long Short-Term Memory (LSTM)
- **Architecture:** Two-layer LSTM
- **Units per layer:** 128
- **Dropout rate:** 0.3
- **Optimiser:** Adam
- **Epochs:** 50 with early stopping (patience=5)
- **Input sequences:** 20 consecutive network flows

### 4. Autoencoder
- **Encoder dimensions:** 41 → 32 → 16 → 8
- **Decoder:** Symmetric (8 → 16 → 32 → 41)
- **Training data:** Normal traffic samples only
- **Threshold:** 95th percentile of reconstruction error on normal traffic

### 5. XGBoost
- **Estimators:** 500
- **Learning rate:** 0.05
- **Max depth:** 6
- **Subsample ratio:** 0.8

---

## Datasets

| Dataset | Records | Features | Attack Categories | Year |
|---------|---------|----------|-------------------|------|
| **NSL-KDD** | 148,517 | 41 | 4 (DoS, Probe, R2L, U2R) | 1999/2009 |
| **CICIDS-2018** | 16,232,943 | 78 | 7 (Brute Force, Web, Infiltration, Bot, DDoS, DoS, Heartbleed) | 2018 |

> Datasets are NOT included in this repository. See [`data/README.md`](data/README.md) for download instructions.  
> After preprocessing: **25 features selected** via RFECV from NSL-KDD; **40 selected** from CICIDS-2018.

---

## Zero Trust Architecture — 5 Pillars

The proposed framework embeds ML-IDS capabilities within each ZTA pillar (NIST SP 800-207):

1. **Identity** — Continuous authentication, UEBA-driven anomaly detection
2. **Devices** — Device trust scoring, endpoint behavioural baselines
3. **Networks** — ML-driven micro-segmentation, east-west traffic analysis
4. **Applications** — API anomaly detection, application-layer threat intelligence
5. **Data** — Autoencoder-based exfiltration detection, DLP integration

See [`docs/ZTA_Framework.md`](docs/ZTA_Framework.md) for the full six-phase implementation roadmap.

---

## Six-Phase Implementation Framework

| Phase | Focus | Key Activities | Timeline |
|-------|-------|----------------|----------|
| **1** | Foundation | ZTA policy design, identity baseline, network inventory | Weeks 1–4 |
| **2** | Data Collection | Deploy network taps, configure SIEM, collect PCAP | Weeks 5–8 |
| **3** | Model Development | Feature engineering, model training, cross-validation | Weeks 9–14 |
| **4** | Integration | Deploy ML engine into SIEM, configure auto-response | Weeks 15–18 |
| **5** | Validation | Red team exercises, penetration testing, tuning | Weeks 19–22 |
| **6** | Operationalisation | SOC playbooks, dashboards, continuous monitoring | Ongoing |

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/ml-ids-zero-trust-cloud.git
cd ml-ids-zero-trust-cloud

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the data pipeline (downloads NSL-KDD and preprocesses)
python src/preprocess.py

# 5. Launch Jupyter notebooks
jupyter notebook notebooks/
```

---

## References

Full APA bibliography as cited in the research paper:

- Apruzzese, G., Colajanni, M., Ferretti, L., & Marchetti, M. (2022). Addressing adversarial attacks against security systems based on machine learning. *Computers and Security, 105*, 102231. https://doi.org/10.1016/j.cose.2021.102231
- Anderson, J. P. (1980). *Computer security threat monitoring and surveillance*. James P. Anderson Co.
- Cloud Security Alliance. (2021). *Security guidance for critical areas of focus in cloud computing v4.0*. https://cloudsecurityalliance.org/research/guidance/
- CrowdStrike. (2024). *CrowdStrike 2024 global threat report*. CrowdStrike Inc.
- Faker, O., & Dogdu, E. (2019). Intrusion detection using big data and deep learning techniques. *Proceedings of the 2019 ACM Southeast Conference*, 86–93. https://doi.org/10.1145/3299815.3314439
- Ferrag, M. A., Maglaras, L., Moschoyiannis, S., & Janicke, H. (2020). Deep learning for cyber security intrusion detection: Approaches, datasets, and comparative study. *Journal of Information Security and Applications, 50*, 102419. https://doi.org/10.1016/j.jisa.2019.102419
- Gartner. (2024). *Gartner forecasts worldwide public cloud end-user spending to reach $679 billion in 2024*. Gartner Inc.
- IBM Security. (2024). *Cost of a data breach report 2024*. IBM Corporation. https://www.ibm.com/reports/data-breach
- Liao, H. J., Lin, C. H. R., Lin, Y. C., & Tung, K. Y. (2013). Intrusion detection system: A comprehensive review. *Journal of Network and Computer Applications, 36*(1), 16–24. https://doi.org/10.1016/j.jnca.2012.09.004
- Liu, H., Lang, B., Liu, M., & Yan, H. (2019). CNN and RNN based payload classification methods for attack detection. *Knowledge-Based Systems, 163*, 332–341. https://doi.org/10.1016/j.knosys.2018.08.036
- Mirsky, Y., Doitshman, T., Elovici, Y., & Shabtai, A. (2018). Kitsune: An ensemble of autoencoders for online network intrusion detection. *Proceedings of the Network and Distributed Systems Security Symposium (NDSS)*. https://doi.org/10.14722/ndss.2018.23235
- National Institute of Standards and Technology. (2018). *Framework for improving critical infrastructure cybersecurity version 1.1*. https://doi.org/10.6028/NIST.CSWP.04162018
- Panaseer. (2023). *Security leaders peer report 2023*. Panaseer Ltd.
- Rose, S., Borchert, O., Mitchell, S., & Connelly, S. (2020). *Zero trust architecture (NIST Special Publication 800-207)*. National Institute of Standards and Technology. https://doi.org/10.6028/NIST.SP.800-207
- Sultana, N., Chilamkurti, N., Peng, W., & Alhadad, R. (2019). Survey on SDN based network intrusion detection system using machine learning approaches. *Peer-to-Peer Networking and Applications, 12*, 493–501. https://doi.org/10.1007/s12083-017-0630-0
- Tang, T., Mhamdi, L., McLernon, D., Zaidi, S. A. R., & Ghogho, M. (2022). Deep recurrent neural network for intrusion detection in SDN-based networks. *IEEE Transactions on Network and Service Management, 19*(3), 2811–2826. https://doi.org/10.1109/TNSM.2022.3143542
- Verizon. (2024). *2024 data breach investigations report*. Verizon Communications Inc.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Research conducted at EC-Council University | MSc Cybersecurity — Cloud Security Architecture | 2026*
