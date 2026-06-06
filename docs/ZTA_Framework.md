# Zero Trust Architecture (ZTA) Framework

**Machine Learning-Based Intrusion Detection for Cloud Network Security: A Zero Trust Architecture Approach**

*Source: Machethe, D.M. (2026). ECCU500: Managing Secure Network Systems. EC-Council University.*

---

## Overview

This document provides a comprehensive description of the Zero Trust Architecture (ZTA) framework and its integration with machine learning-based intrusion detection systems (ML-IDS), as proposed in the research paper (§5.5, Table 4).

The framework is grounded in **NIST Special Publication 800-207** (Rose et al., 2020) and operationalises the guiding principle of *never trust, always verify* through ML-driven continuous monitoring at every ZTA enforcement layer. Zero Trust rejects the traditional perimeter-based security model by assuming that every network request, regardless of origin, must be authenticated, authorised, and continuously validated before access is granted.

Key ZTA tenets (NIST SP 800-207, §2):

- All data sources and computing services are considered resources
- All communication is secured regardless of network location
- Access to individual enterprise resources is granted on a per-session basis
- Access to resources is determined by dynamic policy and observed client identity attributes
- The enterprise monitors and measures the integrity and security posture of all owned and associated assets
- Authentication and authorisation are dynamic and strictly enforced

---

## Zero Trust Architecture: Five Pillars (NIST SP 800-207)

The five-pillar model structures ZTA enforcement zones. ML-IDS capabilities are mapped to each pillar based on the threat surface it protects.

| Pillar | Core Function | Primary Threats | ML Detection Method |
|--------|--------------|-----------------|---------------------|
| **Identity** | Continuous authentication, MFA, UEBA | Credential stuffing, brute force, insider threat | UEBA clustering + LSTM |
| **Devices** | Device trust scoring, endpoint health monitoring | Rogue devices, endpoint compromise | Random Forest classification |
| **Networks** | Micro-segmentation, east-west traffic analysis | Lateral movement, port scanning, DDoS | Random Forest + XGBoost + LSTM |
| **Applications** | API security, application-layer anomaly detection | SQL injection, API abuse, zero-day exploits | Autoencoder + XGBoost |
| **Data** | Data classification, exfiltration prevention | DNS tunnelling, covert channels, DLP bypass | Autoencoder + ensemble methods |

---

## Pillar 1: Identity

### Description

The Identity pillar enforces continuous authentication and behavioural verification of every user and service account. Zero Trust mandates that credentials alone are insufficient; the system must validate context, behaviour patterns, device posture, and risk scores at the time of every access request.

### ML Integration Point

**User and Entity Behaviour Analytics (UEBA)** combined with LSTM temporal modelling detects anomalies in authentication sequences. LSTM ingests 20-timestep sequences of login events and session activity, flagging deviations from established baselines with 98.1% accuracy (paper §6.2.1).

- **Detection capability:** Credential abuse, impossible travel, after-hours access, privilege escalation
- **Model:** LSTMIDS (2-layer, 128 units, dropout=0.3, sequences of 20 events)
- **Output:** Anomaly score fed to ZTA policy engine for dynamic access revocation

### Tools and Technologies

| Category | Tools |
|----------|-------|
| Identity Provider | Azure Active Directory, Okta, AWS IAM Identity Centre |
| MFA | Microsoft Authenticator, Duo Security, FIDO2 keys |
| UEBA Platform | Microsoft Sentinel UEBA, Securonix, Splunk UBA |
| PAM (Privileged Access) | CyberArk, BeyondTrust |
| ML Runtime | TensorFlow/Keras (LSTM), scikit-learn |

---

## Pillar 2: Devices

### Description

The Devices pillar maintains a real-time inventory of all endpoint devices accessing enterprise resources. Trust is not assumed by device type or network location; each device is continuously assessed for compliance, patch level, behavioural anomalies, and integrity attestation.

### ML Integration Point

**Random Forest classification** monitors device telemetry and endpoint logs, building behavioural baselines per device category. Anomalies in resource access patterns, process execution sequences, or network communication profiles trigger trust-score adjustments.

- **Detection capability:** Rogue device onboarding, endpoint compromise, malware persistence, unusual process execution
- **Model:** RandomForestIDS (200 trees, Gini, n_jobs=-1, 10-fold CV accuracy 96.8%)
- **Output:** Trust score updates fed to Network Access Control (NAC) for dynamic policy enforcement

### Tools and Technologies

| Category | Tools |
|----------|-------|
| Endpoint Detection | CrowdStrike Falcon, Microsoft Defender for Endpoint, Carbon Black |
| Mobile Device Management | Microsoft Intune, Jamf Pro, VMware Workspace ONE |
| Network Access Control | Cisco ISE, Aruba ClearPass |
| Vulnerability Management | Tenable.io, Qualys, Rapid7 InsightVM |
| ML Runtime | scikit-learn RandomForestClassifier |

---

## Pillar 3: Networks

### Description

The Networks pillar replaces flat network architectures with dynamic micro-segmentation, granting least-privilege network access per session. East-west (lateral) traffic is fully monitored and policy-controlled. Traditional perimeter firewalls are insufficient; every inter-segment flow is treated as potentially hostile.

### ML Integration Point

This pillar is the primary deployment target for the ML-IDS. Three models operate in ensemble for multi-layer detection across the NSL-KDD attack taxonomy (DoS, Probe, R2L, U2R):

- **Random Forest** (96.8% accuracy): Fast bulk traffic classification for known attack signatures
- **XGBoost** (97.3% accuracy): Gradient boosting for complex multi-feature flow patterns
- **LSTM** (98.1% accuracy): Temporal pattern recognition for sequential attack chains

Network flow features (src_bytes, dst_bytes, count, srv_count, serror_rate, etc.) are extracted at wire speed and fed through the preprocessing pipeline (`src/preprocess.py`) before inference.

### Tools and Technologies

| Category | Tools |
|----------|-------|
| Traffic Capture | Zeek (Bro), Suricata, Arkime (formerly Moloch) |
| SIEM / SOAR | Microsoft Sentinel, Splunk SOAR, IBM QRadar |
| Network Policy | Cisco ACI, VMware NSX-T, AWS Security Groups |
| ML Runtime | scikit-learn, XGBoost, TensorFlow/Keras |
| Feature Extraction | custom `src/preprocess.py` pipeline |

---

## Pillar 4: Applications

### Description

The Applications pillar secures access to applications and APIs. Zero Trust mandates that application-layer traffic is inspected regardless of source network. Cloud-native and SaaS applications expose API endpoints that are frequent targets of credential abuse, injection attacks, and zero-day exploitation.

### ML Integration Point

**Autoencoder anomaly detection** is particularly suited to the Applications pillar because novel attack patterns (zero-days, unknown API abuse) cannot be labelled in advance. The Autoencoder trains exclusively on normal API traffic and flags any request whose reconstruction error exceeds the 95th-percentile threshold.

- **Detection capability:** SQL injection, API parameter tampering, zero-day exploits, abnormal API call sequences
- **Model:** AutoencoderIDS (input_dim -> 32 -> 16 -> 8 -> 16 -> 32 -> input_dim, MSE loss, threshold=95th pct)
- **Complementary model:** XGBoostIDS for known application-layer attack classification
- **Output:** Reconstruction error score and binary anomaly flag forwarded to WAF and SOAR

### Tools and Technologies

| Category | Tools |
|----------|-------|
| Web Application Firewall | AWS WAF, Cloudflare WAF, Azure Application Gateway WAF |
| API Gateway | AWS API Gateway, Kong, Azure APIM |
| Application Security Testing | OWASP ZAP, Burp Suite Pro |
| ML Runtime | TensorFlow/Keras (Autoencoder), XGBoost |
| Anomaly Threshold Storage | joblib (autoencoder_threshold.pkl) |

---

## Pillar 5: Data

### Description

The Data pillar classifies all enterprise data assets and enforces access controls, encryption, and exfiltration prevention at the data layer. Data-centric security applies regardless of where the data resides (on-premises, cloud, or endpoint). Zero Trust at the data layer means that even authenticated users with legitimate access are monitored for abnormal data egress patterns.

### ML Integration Point

**Autoencoder ensemble methods** monitor data access and egress patterns, training on normal data-access baselines and detecting covert exfiltration channels (DNS tunnelling, steganography, encrypted exfiltration over allowed protocols).

- **Detection capability:** Data exfiltration, covert channels, DLP bypass, abnormal bulk downloads
- **Model:** AutoencoderIDS (unsupervised) + ensemble scoring
- **Output:** Exfiltration risk score fed to DLP platform for automated block/quarantine

### Tools and Technologies

| Category | Tools |
|----------|-------|
| Data Loss Prevention | Microsoft Purview, Forcepoint DLP, Symantec DLP |
| Cloud Access Security Broker | Microsoft Defender for Cloud Apps, Netskope, Zscaler CASB |
| Data Classification | Microsoft Information Protection, BigID |
| Encryption | HashiCorp Vault, AWS KMS, Azure Key Vault |
| ML Runtime | TensorFlow/Keras (Autoencoder) |

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
