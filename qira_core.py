
import time
import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

RANDOM_STATE = 42


def load_dataset():
    data = load_breast_cancer(as_frame=True)
    X = data.data.copy()
    # sklearn: malignant=0, benign=1. QIRA: malignant/disease=1.
    y = 1 - data.target.to_numpy()
    return data, X, y


def prepare_data(n_components=4, demo_samples=80, test_size=0.30):
    data, X_df, y = load_dataset()

    # Full benchmark split
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X_df.to_numpy(), y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    scaler_full = StandardScaler()
    X_train_full_s = scaler_full.fit_transform(X_train_full)
    X_test_full_s = scaler_full.transform(X_test_full)
    pca_full = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_train_full_p = pca_full.fit_transform(X_train_full_s)
    X_test_full_p = pca_full.transform(X_test_full_s)

    # Small stratified live-demo subset for quantum simulation
    if demo_samples and demo_samples < len(X_df):
        X_demo, _, y_demo, _ = train_test_split(
            X_df.to_numpy(), y, train_size=demo_samples,
            stratify=y, random_state=RANDOM_STATE
        )
    else:
        X_demo, y_demo = X_df.to_numpy(), y

    X_train_q, X_test_q, y_train_q, y_test_q = train_test_split(
        X_demo, y_demo, test_size=test_size, stratify=y_demo, random_state=RANDOM_STATE
    )
    scaler_q = StandardScaler()
    X_train_q_s = scaler_q.fit_transform(X_train_q)
    X_test_q_s = scaler_q.transform(X_test_q)

    pca_q = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_train_q_p = pca_q.fit_transform(X_train_q_s)
    X_test_q_p = pca_q.transform(X_test_q_s)

    angle_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
    X_train_q_a = angle_scaler.fit_transform(X_train_q_p)
    X_test_q_a = angle_scaler.transform(X_test_q_p)

    return {
        "data": data, "X_df": X_df, "y": y,
        "X_train_full": X_train_full, "X_test_full": X_test_full,
        "y_train_full": y_train_full, "y_test_full": y_test_full,
        "scaler_full": scaler_full, "pca_full": pca_full,
        "X_train_full_p": X_train_full_p, "X_test_full_p": X_test_full_p,
        "X_train_q": X_train_q, "X_test_q": X_test_q,
        "y_train_q": y_train_q, "y_test_q": y_test_q,
        "scaler_q": scaler_q, "pca_q": pca_q,
        "X_train_q_p": X_train_q_p, "X_test_q_p": X_test_q_p,
        "angle_scaler": angle_scaler,
        "X_train_q_a": X_train_q_a, "X_test_q_a": X_test_q_a,
        "demo_samples": len(X_demo)
    }


def classification_metrics(y_true, y_pred, y_prob=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    result = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "False Negative Rate": fnr,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "ROC-AUC": np.nan,
    }
    if y_prob is not None:
        try:
            result["ROC-AUC"] = roc_auc_score(y_true, y_prob)
        except Exception:
            pass
    return result


def _models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=RANDOM_STATE
        )
    return models


def train_classical_models(prepared, use_live_subset=False):
    if use_live_subset:
        X_train, X_test = prepared["X_train_q_p"], prepared["X_test_q_p"]
        y_train, y_test = prepared["y_train_q"], prepared["y_test_q"]
    else:
        X_train, X_test = prepared["X_train_full_p"], prepared["X_test_full_p"]
        y_train, y_test = prepared["y_train_full"], prepared["y_test_full"]

    results, trained = [], {}
    for name, model in _models().items():
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        row = classification_metrics(y_test, pred, prob)
        row["Model"] = name
        row["Training Time (s)"] = elapsed
        results.append(row)
        trained[name] = model

    return trained, pd.DataFrame(results)


class VQCModel:
    def __init__(self, n_qubits=4, n_layers=2, learning_rate=0.12, seed=42):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        np.random.seed(seed)
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="autograd", diff_method="parameter-shift")
        def circuit(weights, x):
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit
        self.weights = pnp.array(
            0.1 * np.random.randn(n_layers, n_qubits, 2), requires_grad=True
        )
        self.bias = pnp.array(0.0, requires_grad=True)
        self.loss_history = []
        self.training_time = 0.0

    def predict_proba_one(self, weights, bias, x):
        z = self.circuit(weights, x) + bias
        return 1.0 / (1.0 + pnp.exp(-z))

    def _cost(self, weights, bias, X, y):
        preds = pnp.array([self.predict_proba_one(weights, bias, x) for x in X])
        eps = 1e-7
        preds = pnp.clip(preds, eps, 1 - eps)
        y_arr = pnp.array(y)
        return -pnp.mean(
            y_arr * pnp.log(preds) + (1 - y_arr) * pnp.log(1 - preds)
        )

    def fit(self, X, y, epochs=10):
        opt = qml.AdamOptimizer(stepsize=self.learning_rate)
        self.loss_history = []
        start = time.time()
        for _ in range(epochs):
            (self.weights, self.bias), loss = opt.step_and_cost(
                lambda w, b: self._cost(w, b, X, y),
                self.weights, self.bias
            )
            self.loss_history.append(float(loss))
        self.training_time = time.time() - start
        return self

    def predict_proba(self, X):
        return np.array([
            float(self.predict_proba_one(self.weights, self.bias, x)) for x in X
        ])

    def draw_circuit(self):
        return qml.draw(self.circuit)(self.weights, np.zeros(self.n_qubits))


def train_vqc(prepared, epochs=10, threshold=0.5):
    model = VQCModel(n_qubits=prepared["X_train_q_a"].shape[1])
    model.fit(prepared["X_train_q_a"], prepared["y_train_q"], epochs=epochs)
    prob = model.predict_proba(prepared["X_test_q_a"])
    pred = (prob >= threshold).astype(int)
    metrics = classification_metrics(prepared["y_test_q"], pred, prob)
    metrics["Model"] = "Quantum VQC"
    metrics["Training Time (s)"] = model.training_time
    return model, metrics, prob


def threshold_metrics(y_true, probabilities, threshold):
    pred = (np.asarray(probabilities) >= threshold).astype(int)
    return classification_metrics(y_true, pred, probabilities)


def threshold_curve(y_true, probabilities, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.10, 0.90, 17)
    rows = []
    for threshold in thresholds:
        row = threshold_metrics(y_true, probabilities, float(threshold))
        rows.append({
            "Threshold": float(threshold),
            "Sensitivity": row["Sensitivity"],
            "Specificity": row["Specificity"],
            "False Negative Rate": row["False Negative Rate"],
            "Accuracy": row["Accuracy"]
        })
    return pd.DataFrame(rows)


def risk_band(probability):
    if probability < 0.30:
        return "LOW"
    if probability <= 0.70:
        return "MODERATE"
    return "HIGH"
