<<<<<<< HEAD
# QIRA - Quantum Integrated Risk Analytics
**Hybrid Quantum-Classical Machine Learning for Early Disease Detection**

QIRA is a prototype that explores the use of **hybrid quantum-classical machine learning** for early disease detection.

### Pipeline

```text
Biomedical Dataset
       ↓
Data Preprocessing
       ↓
StandardScaler
       ↓
PCA
       ↓
Classical ML Model
       ↓
Quantum Feature Processing
       ↓
Prediction & Evaluation
```

### Tech Stack

* Python
* NumPy
* Scikit-learn
* PennyLane
* PCA
* Logistic Regression
* Quantum Machine Learning

### Current Prototype

The current prototype uses the **Breast Cancer Wisconsin dataset** to demonstrate the complete ML pipeline, including preprocessing, dimensionality reduction, classical classification, quantum processing, and model evaluation.

### Future Scope

* Interactive disease-risk dashboard
* Patient data input
* Risk prediction and confidence visualization
* Comparison of classical vs hybrid quantum models
* Model explainability
* Support for additional biomedical datasets

> **Note:** QuantumDx is a research/prototype project and is not intended for clinical diagnosis.
=======
# QIRA Polished

**Quantum-Integrated Risk Analytics** — SIH 26139 prototype for hybrid quantum-classical early disease-risk research.

## Run on macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
```

## Demo flow

1. Click **Prepare / Reset Experiment**
2. Open **Experiment Lab**
3. Click **Run All Models**
4. Present **Classical vs Quantum**
5. Show **Quantum Circuit**
6. Demonstrate **Risk & Threshold**
7. Finish with **Explainability** and **Methodology**

## Key upgrades

- Premium SIH-ready Streamlit interface
- Malignant/disease is the positive class
- Leakage-safe preprocessing
- Full classical benchmark + fair same-subset quantum/classical benchmark
- Logistic Regression, SVM, Random Forest, optional XGBoost
- Real 4-qubit PennyLane VQC on `default.qubit`
- Binary cross-entropy and Adam training
- Sensitivity, specificity, ROC-AUC, F1, FNR
- Training-loss visualization
- Threshold trade-off graph
- Patient-level prototype risk output
- PCA contribution explainability
- Transparent research/clinical guardrails

This project is a research / decision-support prototype and is not clinically validated.
>>>>>>> 3e4a5a3 (Add working QIRA prototype)
