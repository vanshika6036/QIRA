SIH26139 : **Hybrid Quantum-Classical Machine Learning for Early Disease Detection**

# QIRA - Quantum Integrated Risk Analytics
QIRA is a prototype that explores the use of **hybrid quantum-classical machine learning** for early disease detection.

### Dashboard
https://binx53qpj6kzgogz9umh8f.streamlit.app/

### Methidology 
<img width="1038" height="561" alt="image" src="https://github.com/user-attachments/assets/a3b79199-d04d-4901-b2e9-74d5e0f9cee2" />


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

> **Note:** QIRA is a research/prototype project and is not intended for clinical diagnosis.
