
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from qira_core import (
    prepare_data, train_classical_models, train_vqc,
    threshold_metrics, threshold_curve, risk_band
)

st.set_page_config(
    page_title="QIRA | Quantum-Integrated Risk Analytics",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- visual system ----------
st.markdown("""
<style>
:root {
    --qira-cyan: #5DE7FF;
    --qira-violet: #B69CFF;
    --qira-bg: #050B14;
    --qira-panel: #0B1522;
    --qira-card: #101C2B;
    --qira-border: #31445A;
    --qira-text: #FFFFFF;
    --qira-soft: #DDE8F3;
    --qira-muted: #C7D5E3;
}

/* ---------- APP BACKGROUND ---------- */
html, body, [class*="css"] {
    color: var(--qira-text) !important;
}

.stApp {
    background:
      radial-gradient(circle at 18% 0%, rgba(93,231,255,.10), transparent 24%),
      radial-gradient(circle at 86% 12%, rgba(182,156,255,.10), transparent 24%),
      linear-gradient(180deg, #050B14 0%, #08111D 100%);
    color: var(--qira-text) !important;
}

.block-container {
    padding-top: 1.55rem;
    max-width: 1500px;
}

/* ---------- GLOBAL TEXT ---------- */
.stApp p,
.stApp span,
.stApp div,
.stApp label,
.stApp li,
.stApp small,
.stApp strong,
.stApp b {
    color: var(--qira-text) !important;
}

.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6 {
    color: #FFFFFF !important;
    opacity: 1 !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
    color: #F7FBFF !important;
    opacity: 1 !important;
}

/* ---------- SIDEBAR ---------- */
[data-testid="stSidebar"] {
    background: #07111D !important;
    border-right: 1px solid #2C4056 !important;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
    opacity: 1 !important;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #FFFFFF !important;
}

/* ---------- WIDGET LABELS ---------- */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}

[data-baseweb="radio"] label,
[data-baseweb="radio"] span,
[data-baseweb="slider"] *,
[data-baseweb="select"] * {
    color: #FFFFFF !important;
    opacity: 1 !important;
}

/* ---------- CAPTIONS / SECONDARY TEXT ---------- */
.stCaptionContainer,
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p,
[data-testid="stCaptionContainer"] span {
    color: #D7E3EF !important;
    opacity: 1 !important;
}

/* ---------- METRICS ---------- */
div[data-testid="stMetric"] {
    background: #0F1A29 !important;
    border: 1px solid #334A63 !important;
    padding: 14px !important;
    border-radius: 17px !important;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] span {
    color: #D6E3EF !important;
    opacity: 1 !important;
    font-weight: 700 !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {
    color: #FFFFFF !important;
    opacity: 1 !important;
}

/* ---------- BUTTONS ---------- */
.stButton > button {
    background: #102237 !important;
    color: #FFFFFF !important;
    border-radius: 12px !important;
    border: 1px solid #5DE7FF !important;
    min-height: 42px !important;
    font-weight: 800 !important;
}

.stButton > button:hover {
    background: #17314F !important;
    border-color: #9FEFFF !important;
    color: #FFFFFF !important;
}

/* ---------- INPUTS ---------- */
input,
textarea,
[data-baseweb="input"] input {
    color: #FFFFFF !important;
    background: #0E1927 !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

[data-baseweb="select"] > div,
[data-baseweb="popover"] {
    background: #0E1927 !important;
    color: #FFFFFF !important;
}

/* ---------- TABLE / DATAFRAME ---------- */
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] * {
    color: #FFFFFF !important;
    opacity: 1 !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid #31445A !important;
    border-radius: 12px !important;
}

/* ---------- STATUS / INFO / WARNING ---------- */
[data-testid="stAlert"],
[data-testid="stAlert"] *,
[data-testid="stStatusWidget"],
[data-testid="stStatusWidget"] *,
[data-testid="stExpander"],
[data-testid="stExpander"] * {
    color: #FFFFFF !important;
    opacity: 1 !important;
}

/* ---------- CODE ---------- */
code,
pre,
[data-testid="stCodeBlock"] * {
    color: #EAFBFF !important;
    opacity: 1 !important;
}

/* ---------- CUSTOM COMPONENTS ---------- */
.qira-hero {
    position: relative;
    overflow: hidden;
    padding: 34px 34px 30px 34px;
    border: 1px solid #35506A;
    border-radius: 24px;
    background: linear-gradient(135deg,#0E2234,#151D35);
    box-shadow: 0 14px 50px rgba(0,0,0,.35);
    margin-bottom: 18px;
}

.qira-hero:after {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    right: -90px;
    top: -110px;
    border-radius: 999px;
    background: rgba(93,231,255,.12);
}

.qira-kicker {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    border: 1px solid #5DE7FF;
    background: #0D2533;
    color: #CFF8FF !important;
    font-size: .78rem;
    font-weight: 800;
    letter-spacing: .08em;
}

.qira-hero h1 {
    font-size: 3.1rem;
    line-height: 1;
    margin: 14px 0 8px 0;
    color: #FFFFFF !important;
}

.qira-hero p {
    color: #E7F0F8 !important;
    font-size: 1.05rem;
    max-width: 820px;
    margin: 0;
    opacity: 1 !important;
}

.qira-card {
    padding: 18px;
    border-radius: 18px;
    background: #0F1A29;
    border: 1px solid #334A63;
    height: 100%;
    color: #FFFFFF !important;
}

.qira-card p,
.qira-card h4,
.qira-card span {
    color: #FFFFFF !important;
    opacity: 1 !important;
}

.qira-note {
    padding: 14px 16px;
    border-left: 4px solid #5DE7FF;
    border-top: 1px solid #29445A;
    border-right: 1px solid #29445A;
    border-bottom: 1px solid #29445A;
    background: #0D1A27;
    border-radius: 10px;
    color: #F2F7FC !important;
    font-weight: 500;
}

.qira-note * {
    color: #F2F7FC !important;
    opacity: 1 !important;
}

.qira-badge {
    display: inline-block;
    padding: 5px 9px;
    margin-right: 6px;
    border-radius: 999px;
    background: #181D36;
    border: 1px solid #9F89FF;
    color: #EFEAFF !important;
    font-size: .77rem;
    font-weight: 800;
}

.pipeline {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    margin-top: 10px;
}

.pipe-node {
    padding: 9px 12px;
    border: 1px solid #3B526C;
    border-radius: 12px;
    background: #0E1A28;
    color: #FFFFFF !important;
    font-size: .88rem;
    font-weight: 700;
}

.pipe-arrow {
    color: #BFD0E0 !important;
    font-weight: 900;
}

/* Remove any dimming from disabled-like text */
[aria-disabled="true"],
[disabled] {
    opacity: 1 !important;
}

/* Horizontal rules */
hr {
    border-color: #2F435A !important;
}

/* ---------- PLOTLY SVG TEXT ---------- */
.js-plotly-plot .plotly text {
    fill: #FFFFFF !important;
}

.js-plotly-plot .gtitle {
    fill: #FFFFFF !important;
}

</style>
""", unsafe_allow_html=True)

PLOT_CONFIG = {"displayModeBar": False, "responsive": True}

for key, value in {
    "prepared": None,
    "classical_models_full": None,
    "classical_results_full": None,
    "classical_models_live": None,
    "classical_results_live": None,
    "vqc_model": None,
    "vqc_metrics": None,
    "vqc_prob": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_results():
    for key in [
        "classical_models_full","classical_results_full",
        "classical_models_live","classical_results_live",
        "vqc_model","vqc_metrics","vqc_prob"
    ]:
        st.session_state[key] = None


def prepare(demo_samples):
    st.session_state.prepared = prepare_data(4, demo_samples, 0.30)
    reset_results()


def ensure_prepared():
    if st.session_state.prepared is None:
        st.info("Prepare an experiment from the sidebar to continue.")
        st.stop()


def metric_table(df):
    show = df.copy()
    pct_cols = ["Accuracy","Precision","Sensitivity","Specificity","F1","ROC-AUC","False Negative Rate"]
    for col in pct_cols:
        if col in show.columns:
            show[col] = show[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
    if "Training Time (s)" in show.columns:
        show["Training Time (s)"] = show["Training Time (s)"].map(lambda x: f"{x:.3f}")
    return show


# ---------- sidebar ----------
with st.sidebar:
    st.markdown("## ⚛️ QIRA")
    st.caption("Quantum-Integrated Risk Analytics")
    st.markdown(
        '<span class="qira-badge">SIH 26139</span><span class="qira-badge">MedTech</span>',
        unsafe_allow_html=True
    )
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Dataset & PCA",
            "Experiment Lab",
            "Classical vs Quantum",
            "Quantum Circuit",
            "Risk & Threshold",
            "Explainability",
            "Methodology",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### Experiment controls")
    demo_samples = st.select_slider(
        "Quantum demo patients",
        options=[80,120,160,200],
        value=80
    )
    epochs = st.slider("VQC epochs", 5, 30, 10, 5)
    threshold = st.slider("Decision threshold", 0.10, 0.90, 0.50, 0.05)

    if st.button("Prepare / Reset Experiment", use_container_width=True):
        prepare(demo_samples)
        st.success("Experiment ready.")

    st.divider()
    status = "Ready" if st.session_state.prepared is not None else "Waiting"
    st.markdown(f"**System status:** `{status}`")
    st.caption("Quantum backend: PennyLane `default.qubit`")


# ---------- hero ----------
st.markdown("""
<div class="qira-hero">
  <span class="qira-kicker">HYBRID QUANTUM–CLASSICAL CLINICAL AI PROTOTYPE</span>
  <h1>QIRA</h1>
  <p>
    Early disease-risk intelligence that benchmarks quantum-enhanced learning
    against classical machine learning under a transparent, reproducible workflow.
  </p>
</div>
""", unsafe_allow_html=True)


# ---------- pages ----------
if page == "Overview":
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Dataset","Breast Cancer Wisconsin")
    c2.metric("Patients","569")
    c3.metric("Clinical Features","30")
    c4.metric("Quantum Inputs","4")
    c5.metric("Quantum Backend","4 Qubits")

    st.markdown("### One protocol. Two learning paradigms.")
    st.markdown("""
    <div class="pipeline">
      <span class="pipe-node">Biomedical Data</span><span class="pipe-arrow">→</span>
      <span class="pipe-node">StandardScaler</span><span class="pipe-arrow">→</span>
      <span class="pipe-node">PCA 30 → 4</span><span class="pipe-arrow">→</span>
      <span class="pipe-node">Classical Models</span>
      <span class="pipe-node">Angle Encoding + VQC</span><span class="pipe-arrow">→</span>
      <span class="pipe-node">Benchmark</span><span class="pipe-arrow">→</span>
      <span class="pipe-node">Risk + Threshold</span>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    left, mid, right = st.columns(3)
    with left:
        st.markdown("""
        <div class="qira-card">
        <h4>01 · Clinically relevant metrics</h4>
        <p>QIRA goes beyond raw accuracy with sensitivity, specificity,
        false-negative rate, F1 and ROC-AUC.</p>
        </div>
        """, unsafe_allow_html=True)
    with mid:
        st.markdown("""
        <div class="qira-card">
        <h4>02 · Real quantum execution</h4>
        <p>A real PennyLane variational quantum circuit runs locally on
        <code>default.qubit</code>; results are not hard-coded.</p>
        </div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown("""
        <div class="qira-card">
        <h4>03 · No forced quantum advantage</h4>
        <p>The platform reports measured outcomes honestly—even when a
        classical baseline performs better.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown("""
    <div class="qira-note">
    <b>Research scope:</b> QIRA is a decision-support / research prototype.
    It is not a clinically validated diagnostic device and does not replace a clinician.
    </div>
    """, unsafe_allow_html=True)

elif page == "Dataset & PCA":
    ensure_prepared()
    p = st.session_state.prepared
    df = p["X_df"].copy()
    df["Clinical Label"] = np.where(p["y"] == 1, "Malignant", "Benign")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total patients", len(df))
    c2.metric("Original features", p["X_df"].shape[1])
    c3.metric("PCA components", 4)
    c4.metric("Variance retained", f"{p['pca_full'].explained_variance_ratio_.sum():.2%}")

    left, right = st.columns([1,1])
    with left:
        counts = df["Clinical Label"].value_counts().reset_index()
        counts.columns = ["Class","Patients"]
        fig = px.bar(counts, x="Class", y="Patients", title="Class distribution")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
    with right:
        var = p["pca_full"].explained_variance_ratio_
        cum = np.cumsum(var)
        fig = go.Figure()
        fig.add_bar(x=[f"PC{i+1}" for i in range(4)], y=var, name="Individual variance")
        fig.add_scatter(x=[f"PC{i+1}" for i in range(4)], y=cum, name="Cumulative variance", mode="lines+markers")
        fig.update_layout(
            title="PCA information retained", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis_tickformat=".0%"
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

    st.markdown("### Biomedical feature preview")
    st.dataframe(df.head(12), use_container_width=True, hide_index=True)

elif page == "Experiment Lab":
    ensure_prepared()
    p = st.session_state.prepared

    st.markdown("### Train the complete experiment")
    c1,c2,c3 = st.columns([1,1,1])
    with c1:
        st.metric("Quantum train samples", len(p["y_train_q"]))
    with c2:
        st.metric("Quantum test samples", len(p["y_test_q"]))
    with c3:
        st.metric("VQC epochs", epochs)

    if st.button("▶ Run All Models", use_container_width=True, type="primary"):
        with st.status("Running QIRA experiment...", expanded=True) as status:
            st.write("Training full-dataset classical benchmark...")
            mf, rf = train_classical_models(p, use_live_subset=False)
            st.session_state.classical_models_full = mf
            st.session_state.classical_results_full = rf

            st.write("Training fair live-subset classical benchmark...")
            ml, rl = train_classical_models(p, use_live_subset=True)
            st.session_state.classical_models_live = ml
            st.session_state.classical_results_live = rl

            st.write("Training 4-qubit VQC...")
            qmodel, qmetrics, qprob = train_vqc(p, epochs=epochs, threshold=threshold)
            st.session_state.vqc_model = qmodel
            st.session_state.vqc_metrics = qmetrics
            st.session_state.vqc_prob = qprob
            status.update(label="Experiment complete", state="complete")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Full classical benchmark")
        if st.session_state.classical_results_full is None:
            st.caption("Run the experiment to populate results.")
        else:
            cols = ["Model","Accuracy","Sensitivity","Specificity","F1","ROC-AUC","Training Time (s)"]
            st.dataframe(metric_table(st.session_state.classical_results_full[cols]), use_container_width=True, hide_index=True)

    with right:
        st.markdown("#### Quantum VQC")
        if st.session_state.vqc_metrics is None:
            st.caption("Run the experiment to populate results.")
        else:
            m = st.session_state.vqc_metrics
            a,b,c = st.columns(3)
            a.metric("Accuracy",f"{m['Accuracy']:.2%}")
            b.metric("Sensitivity",f"{m['Sensitivity']:.2%}")
            c.metric("Specificity",f"{m['Specificity']:.2%}")
            d,e = st.columns(2)
            d.metric("ROC-AUC",f"{m['ROC-AUC']:.3f}")
            e.metric("Training time",f"{m['Training Time (s)']:.2f}s")

            loss_df = pd.DataFrame({
                "Epoch":range(1,len(st.session_state.vqc_model.loss_history)+1),
                "Loss":st.session_state.vqc_model.loss_history
            })
            fig = px.line(loss_df,x="Epoch",y="Loss",markers=True,title="VQC training loss")
            fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig,use_container_width=True,config=PLOT_CONFIG)

elif page == "Classical vs Quantum":
    ensure_prepared()
    if st.session_state.classical_results_live is None or st.session_state.vqc_metrics is None:
        st.info("Run **Experiment Lab → Run All Models** first.")
        st.stop()

    live = st.session_state.classical_results_live.copy()
    combined = pd.concat([live, pd.DataFrame([st.session_state.vqc_metrics])], ignore_index=True)

    st.markdown("### Fair live-subset comparison")
    st.markdown("""
    <div class="qira-note">
    Every model in this table is evaluated on the same live-demo patient subset and the same
    four PCA features. This is the fairest direct classical-vs-quantum comparison in the prototype.
    </div>
    """, unsafe_allow_html=True)

    cols = ["Model","Accuracy","Precision","Sensitivity","Specificity","F1","ROC-AUC","False Negative Rate","Training Time (s)"]
    st.dataframe(metric_table(combined[cols]), use_container_width=True, hide_index=True)

    metrics = ["Accuracy","Sensitivity","Specificity","F1","ROC-AUC"]
    long_df = combined.melt(id_vars=["Model"],value_vars=metrics,var_name="Metric",value_name="Score")
    fig = px.bar(long_df,x="Metric",y="Score",color="Model",barmode="group",title="Performance benchmark")
    fig.update_layout(
        template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0,1.05],tickformat=".0%")
    )
    st.plotly_chart(fig,use_container_width=True,config=PLOT_CONFIG)

    best = combined.loc[combined["Accuracy"].idxmax()]
    st.caption(
        f"Highest live-subset accuracy in this run: **{best['Model']} ({best['Accuracy']:.2%})**. "
        "QIRA does not interpret this small demo run as clinical superiority."
    )

elif page == "Quantum Circuit":
    ensure_prepared()
    if st.session_state.vqc_model is None:
        st.info("Run the VQC from **Experiment Lab** first.")
        st.stop()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Qubits","4")
    c2.metric("Encoding","Angle / RY")
    c3.metric("Variational layers","2")
    c4.metric("Measurement","Pauli-Z")

    st.markdown("### Executed circuit")
    st.code(st.session_state.vqc_model.draw_circuit(), language=None)

    st.markdown("""
    **How to explain this to judges:** each PCA component becomes a rotation angle on one qubit.
    Trainable `RY` and `RZ` gates change the quantum state, `CNOT` gates couple neighbouring qubits,
    and the final Pauli-Z expectation is converted to a malignancy probability.
    """)

elif page == "Risk & Threshold":
    ensure_prepared()
    if st.session_state.vqc_prob is None:
        st.info("Run the VQC from **Experiment Lab** first.")
        st.stop()

    p = st.session_state.prepared
    m = threshold_metrics(p["y_test_q"], st.session_state.vqc_prob, threshold)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Decision threshold",f"{threshold:.2f}")
    c2.metric("Sensitivity",f"{m['Sensitivity']:.2%}")
    c3.metric("Specificity",f"{m['Specificity']:.2%}")
    c4.metric("False-negative rate",f"{m['False Negative Rate']:.2%}")

    curve = threshold_curve(p["y_test_q"], st.session_state.vqc_prob)
    curve_long = curve.melt(
        id_vars=["Threshold"],
        value_vars=["Sensitivity","Specificity","False Negative Rate"],
        var_name="Metric",value_name="Score"
    )
    fig = px.line(curve_long,x="Threshold",y="Score",color="Metric",markers=True,title="Threshold trade-off")
    fig.add_vline(x=threshold,line_dash="dash")
    fig.update_layout(
        template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        yaxis_tickformat=".0%"
    )
    st.plotly_chart(fig,use_container_width=True,config=PLOT_CONFIG)

    st.markdown("### Patient-level risk view")
    idx = st.slider("Select test patient",0,len(st.session_state.vqc_prob)-1,0)
    prob = float(st.session_state.vqc_prob[idx])
    true_label = int(p["y_test_q"][idx])
    band = risk_band(prob)

    left,right = st.columns([1,2])
    with left:
        st.metric("Malignancy probability",f"{prob:.2%}")
        st.progress(min(max(prob,0.0),1.0))
        st.markdown(f"**Prototype risk band:** `{band}`")
        st.markdown("**Model decision:** " + ("Higher-risk / malignant" if prob >= threshold else "Lower-risk / benign"))
        st.markdown("**Known test label:** " + ("Malignant" if true_label == 1 else "Benign"))
    with right:
        st.markdown("""
        <div class="qira-card">
        <h4>Why threshold tuning matters</h4>
        <p>Lowering the threshold can raise sensitivity and reduce missed malignant cases,
        but it may also reduce specificity and create more false positives. QIRA exposes this
        trade-off instead of hiding it behind a single default threshold.</p>
        </div>
        """, unsafe_allow_html=True)

    st.warning("The Low / Moderate / High bands are prototype research bands, not medically validated cut-offs.")

elif page == "Explainability":
    ensure_prepared()
    p = st.session_state.prepared

    st.markdown("### PCA contribution map")
    loadings = pd.DataFrame(
        p["pca_full"].components_.T,
        index=p["X_df"].columns,
        columns=[f"PC{i+1}" for i in range(4)]
    )
    st.dataframe(loadings.round(3),use_container_width=True)

    abs_load = loadings.abs().max(axis=1).sort_values(ascending=False).head(10)
    fig = px.bar(
        x=abs_load.values, y=abs_load.index, orientation="h",
        labels={"x":"Max absolute PCA loading","y":"Original feature"},
        title="Top original features influencing PCA space"
    )
    fig.update_layout(
        template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        yaxis={"categoryorder":"total ascending"}
    )
    st.plotly_chart(fig,use_container_width=True,config=PLOT_CONFIG)

    if st.session_state.classical_models_full and "Random Forest" in st.session_state.classical_models_full:
        rf = st.session_state.classical_models_full["Random Forest"]
        imp = pd.DataFrame({
            "PCA Component":[f"PC{i+1}" for i in range(4)],
            "Importance":rf.feature_importances_
        }).sort_values("Importance",ascending=True)
        fig = px.bar(imp,x="Importance",y="PCA Component",orientation="h",title="Random Forest importance over PCA components")
        fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,use_container_width=True,config=PLOT_CONFIG)

    st.markdown("""
    <div class="qira-note">
    QIRA deliberately separates classical feature explainability from quantum circuit inspection.
    It does not claim that classical SHAP-style explanations directly reveal an internal quantum state.
    </div>
    """, unsafe_allow_html=True)

elif page == "Methodology":
    st.markdown("### Experimental methodology")
    rows = [
        ("1","Data","Breast Cancer Wisconsin diagnostic dataset"),
        ("2","Preprocessing","Train-only StandardScaler fitting to avoid leakage"),
        ("3","Compression","PCA maps 30 clinical features to 4 components"),
        ("4","Classical branch","Logistic Regression, SVM, Random Forest, XGBoost when installed"),
        ("5","Quantum branch","4-qubit PennyLane VQC on default.qubit with Angle Encoding"),
        ("6","Training","Binary cross-entropy + Adam optimization"),
        ("7","Evaluation","Accuracy, precision, sensitivity, specificity, F1, ROC-AUC, FNR"),
        ("8","Decision support","Adjustable classification threshold + research-only risk bands"),
    ]
    st.dataframe(pd.DataFrame(rows,columns=["Step","Layer","Implementation"]),use_container_width=True,hide_index=True)

    st.markdown("### What QIRA is testing")
    st.markdown("""
    > **Research question:** can a compact variational quantum model learn useful structure from a
    PCA-reduced biomedical feature space, and how does that measured performance compare with
    classical baselines under the same evaluation protocol?
    """)

    st.markdown("### Guardrails")
    st.markdown("""
    - No fabricated quantum advantage.
    - No hard-coded model metrics.
    - `default.qubit` is a simulator, not physical quantum hardware.
    - Small-subset live results are demonstration results, not clinical validation.
    - The platform is designed for transparent experimentation and decision-support research.
    """)
