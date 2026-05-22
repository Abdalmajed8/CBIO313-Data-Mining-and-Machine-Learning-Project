import pickle
import numpy as np
import streamlit as st

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetesIQ — Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0D0F14; }
[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] { display: none; }
h1 { font-size: 2.2rem !important; font-weight: 800 !important; }
.result-box {
    padding: 1.5rem;
    border-radius: 14px;
    margin: 1rem 0;
    border: 1px solid;
}
.green-box  { background: rgba(34,197,94,.1);  border-color: rgba(34,197,94,.3);  }
.amber-box  { background: rgba(245,158,11,.1); border-color: rgba(245,158,11,.3); }
.red-box    { background: rgba(239,68,68,.1);  border-color: rgba(239,68,68,.3);  }
</style>
""", unsafe_allow_html=True)

# ── Load model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ── Scaler stats from the 100k-row diabetes dataset ─────────
MEANS = np.array([41.89, 27.32, 5.53, 138.06])   # age, bmi, hbA1c, glucose
STDS  = np.array([22.52,  6.64,  1.07,  40.71])

def scale(vals):
    return (np.array(vals) - MEANS) / STDS

# ── UI ───────────────────────────────────────────────────────
st.title("🩺 DiabetesIQ")
st.markdown("#### Diabetes Risk Predictor")
st.markdown("Enter your clinical measurements and the model will assess your risk profile.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    age     = st.slider("Age (years)",          1,   100,  35)
    hba1c   = st.slider("HbA1c Level (%)",      3.0, 15.0, 5.5, step=0.1,
                         help="Normal < 5.7% | Pre-diabetes 5.7–6.4% | Diabetes ≥ 6.5%")

with col2:
    bmi     = st.slider("BMI (kg/m²)",          10.0, 60.0, 25.0, step=0.1,
                         help="Normal 18.5–24.9 | Overweight 25–29.9 | Obese ≥ 30")
    glucose = st.slider("Blood Glucose (mg/dL)", 50,  400,  100,
                         help="Normal 70–99 | Pre-diabetes 100–125 | Diabetes ≥ 126")

st.divider()

# ── Reference table ──────────────────────────────────────────
with st.expander("📊 Clinical reference ranges"):
    st.markdown("""
| Marker | Normal | Pre-Diabetes | Diabetes |
|--------|--------|--------------|----------|
| HbA1c | < 5.7% | 5.7 – 6.4% | ≥ 6.5% |
| Blood Glucose | 70–99 mg/dL | 100–125 mg/dL | ≥ 126 mg/dL |
| BMI | 18.5–24.9 | 25–29.9 (overweight) | ≥ 30 (obese) |
""")

# ── Predict ──────────────────────────────────────────────────
if st.button("🔍 Analyze Risk Profile", use_container_width=True, type="primary"):
    scaled = scale([age, bmi, hba1c, glucose]).reshape(1, -1)
    pred   = int(model.predict(scaled)[0])
    proba  = model.predict_proba(scaled)[0]

    labels = {0: "No Diabetes", 1: "Pre-Diabetes", 2: "Diabetes"}
    icons  = {0: "✅", 1: "⚠️", 2: "🔴"}
    styles = {0: "green-box", 1: "amber-box", 2: "red-box"}
    advice = {
        0: "Your values are within the healthy range. Keep up the good habits!",
        1: "You may be at elevated risk. Consider lifestyle changes and consulting a doctor.",
        2: "High risk indicators detected. Please seek medical advice promptly.",
    }

    # Result banner
    st.markdown(f"""
    <div class="result-box {styles[pred]}">
        <h2 style="margin:0">{icons[pred]} {labels[pred]}</h2>
        <p style="margin:.5rem 0 0; opacity:.8">{advice[pred]}</p>
    </div>
    """, unsafe_allow_html=True)

    # Probability bars
    st.markdown("#### Prediction Probabilities")
    bar_labels = ["No Diabetes", "Pre-Diabetes", "Diabetes"]
    bar_colors = ["normal", "normal", "normal"]
    for i, (lbl, p) in enumerate(zip(bar_labels, proba)):
        st.progress(float(p), text=f"{lbl}: **{p*100:.1f}%**")

    # Input summary
    st.markdown("#### Your Input Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Age",     f"{age} yrs")
    c2.metric("BMI",     f"{bmi:.1f}")
    c3.metric("HbA1c",   f"{hba1c:.1f}%")
    c4.metric("Glucose", f"{glucose} mg/dL")

    # Disclaimer
    st.info("⚠️ **Medical Disclaimer:** This tool is for educational and research purposes only. It does not constitute medical advice. Always consult a qualified healthcare professional.")

