# 🛡️ ML-Augmented FMEA for Aerospace Safety Engineering

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikitlearn)
![Domain](https://img.shields.io/badge/Domain-Safety%20Engineering%20%7C%20NLP-lightgrey)
![Standard](https://img.shields.io/badge/Standard-ARP4761%20%7C%20MIL--STD--1629A-red)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> **NLP + ML pipeline** that automates Failure Mode and Effects Analysis (FMEA) per **ARP4761 / MIL-STD-1629A** — classifying failure modes from free-text incident reports with **>95% accuracy** and predicting Risk Priority Numbers (RPN).

---

## 🎯 Motivation

Traditional FMEA is a manual, time-intensive process: safety engineers review hundreds of incident reports, classify failure modes, and assign Severity / Occurrence / Detectability scores subjectively. This bottleneck slows down design reviews, MRO workflows, and certification processes.

This project demonstrates how **NLP and supervised ML** can automate the classification and prioritization steps — reducing manual effort while maintaining full traceability. The approach is directly applicable to aerospace programs following **ARP4761** (civil) and **MIL-STD-1629A** (military) safety standards.

---

## 📐 Problem Definition

**Input:** Free-text incident report (as written by a technician, pilot, or BITE system)  
**Output:**

| Target | Type | Description |
|---|---|---|
| `system` | Classification (7 classes) | Affected aircraft system |
| `failure_mode` | Classification (28 classes) | Specific failure mode |
| `S` | Regression (1–10) | Severity score |
| `O` | Regression (1–10) | Occurrence score |
| `D` | Regression (1–10) | Detectability score |
| `RPN` | Regression | Risk Priority Number = S × O × D |

---

## 🗂️ Project Structure

```
fmea-ml-aerospace/
│
├── fmea_ml_risk.ipynb     # Main notebook (complete pipeline)
├── requirements.txt
└── README.md
```

---

## 🔬 Methodology

### 1. Dataset
- 2,500 synthetic incident records based on **NASA ASRS** report language patterns
- ARP4761 system taxonomy: Hydraulic, Electrical, Avionics, Fuel, Flight Control, Landing Gear, Engine
- 28 failure modes per ARP4761 / MIL-STD-1629A classification
- SOD scores derived from ARP4761 severity categories and flight phase heuristics

### 2. NLP Pipeline
- **TF-IDF vectorization** with unigrams + bigrams (`max_features=300`)
- Captures domain-specific phrases: "compressor stall", "pressure drop", "sensor drift"
- `sublinear_tf=True` for log normalization — standard in technical text classification

### 3. Classification — System & Failure Mode

| Model | System Accuracy | Failure Mode Accuracy |
|---|---|---|
| Logistic Regression | 88.4% | 79.2% |
| **Random Forest** | **>95%** | **>95%** |

### 4. RPN Component Prediction

| Target | R² | MAE |
|---|---|---|
| S (Severity) | 0.952 | 0.48 |
| O (Occurrence) | 0.747 | 0.91 |
| D (Detectability) | — | 0.82 |
| **RPN** | **0.546** | **28.1** |

> Note: D is difficult to predict from text alone (it depends on system design, not incident description) — this is an expected and interpretable result, consistent with the FMEA literature.

### 5. Risk Matrix
ARP4761-style Severity vs Occurrence matrix with automated risk level assignment:

| S × O | Risk Level |
|---|---|
| ≥ 50 | 🔴 Critical |
| ≥ 25 | 🟠 High |
| ≥ 10 | 🟡 Medium |
| < 10 | 🟢 Low |

---

## 🚀 How to Run

```bash
git clone https://github.com/YOUR_USERNAME/fmea-ml-aerospace
cd fmea-ml-aerospace
pip install -r requirements.txt
jupyter notebook fmea_ml_risk.ipynb
```

### Live Inference Example

```python
classify_incident(
    description="crew observed hydraulic fluid dripping from wing root area",
    flight_phase="Landing"
)
# → System: Hydraulic | Failure mode: Leakage | S=8 O=6 D=7 → RPN=336 | Risk: Critical
```

---

## 🔭 Extensions & Future Work

- **LLM fine-tuning**: use a domain-adapted language model (e.g. fine-tuned BERT on ASRS corpus) for better generalization to unseen report styles
- **Active learning**: engineer feedback loop — model flags uncertain predictions for human review, improving over time
- **Conformal Prediction**: uncertainty quantification on RPN estimates — critical for DO-178C / ARP4754A certification contexts
- **Real ASRS integration**: pipeline to ingest and process NASA ASRS public database directly

---

## 📚 References

- SAE ARP4761: *Guidelines and Methods for Conducting the Safety Assessment Process on Civil Airborne Systems*
- MIL-STD-1629A: *Procedures for Performing a Failure Mode, Effects and Criticality Analysis*
- NASA Aviation Safety Reporting System (ASRS): https://asrs.arc.nasa.gov/
- Drury, C.G. & Gramopadhye, A. (1991). *Training for Visual Inspection*. FAA.

---

## 👤 Author

**[Your Name]** — MSc Space Engineering  
Background in helicopter flight simulation | Learning ML/AI for aerospace applications  
[LinkedIn](#) · [GitHub](#)
