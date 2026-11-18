import io
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ecg_utils import MODEL_FS, bandpass_filter, detect_r_peaks, segment_beats, to_model_rate

MODEL_PATH = "ecg_random_forest_baseline.joblib"
COLORS = {"Normal": "tab:green", "Supraventricular": "tab:orange", "Ventricular": "tab:red"}

st.set_page_config(page_title="AI ECG Monitor", page_icon="🫀", layout="wide")
st.title("🫀 AI-Based ECG Monitoring & Abnormality Detection")
st.warning("Research/education demo only. NOT a medical device; do not use for diagnosis or treatment decisions.")


@st.cache_resource
def load_artifact():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner="Downloading record from PhysioNet...")
def load_mitbih(record_id):
    import wfdb
    rec = wfdb.rdrecord(record_id, pn_dir="mitdb", sampto=360 * 60)  # first 60 s
    return rec.p_signal[:, 0], rec.fs


artifact = load_artifact()
if artifact is None:
    st.error(f"`{MODEL_PATH}` not found. Run the Colab notebook's save cell, download the file, "
             "and place it next to app.py.")
    st.stop()

model = artifact["model"]
pre_s, post_s = artifact["pre_seconds"], artifact["post_seconds"]

# ---------------- Sidebar: input ----------------
st.sidebar.header("ECG input")
source = st.sidebar.radio("Source", ["MIT-BIH demo record", "Upload CSV"])
signal, fs = None, None

if source == "MIT-BIH demo record":
    rid = st.sidebar.selectbox("Record", ["100", "119", "200", "208", "233"], index=2)
    try:
        signal, fs = load_mitbih(rid)
    except Exception as e:
        st.error(f"Could not download record (internet needed): {e}")
else:
    up = st.sidebar.file_uploader("CSV with an ECG column", type=["csv", "txt"])
    fs = st.sidebar.number_input("Sampling rate (Hz)", 50, 2000, 360)
    if up is not None:
        df = pd.read_csv(io.BytesIO(up.read()))
        num_cols = df.select_dtypes("number").columns.tolist()
        if not num_cols:  # maybe headerless
            up.seek(0)
            df = pd.read_csv(up, header=None)
            num_cols = df.select_dtypes("number").columns.tolist()
        col = st.sidebar.selectbox("ECG column", num_cols)
        signal = df[col].dropna().to_numpy(dtype=float)

if signal is None:
    st.info("Choose a demo record or upload a CSV to begin.")
    st.stop()

# ---------------- Processing ----------------
sig = to_model_rate(signal, fs)
filtered = bandpass_filter(sig, MODEL_FS)
peaks = detect_r_peaks(filtered)
X, kept = segment_beats(filtered, peaks, pre_s, post_s)

if len(X) == 0:
    st.error("No beats detected. Check the sampling rate and that the signal is a raw ECG.")
    st.stop()

pred = model.predict(X)
proba = model.predict_proba(X).max(axis=1)
result = pd.DataFrame({"time_s": kept / MODEL_FS, "predicted_class": pred, "confidence": proba.round(3)})

# ---------------- Summary ----------------
duration = len(sig) / MODEL_FS
hr = 60 * len(kept) / duration
counts = result["predicted_class"].value_counts()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Duration", f"{duration:.1f} s")
c2.metric("Beats detected", len(kept))
c3.metric("Avg heart rate", f"{hr:.0f} bpm")
c4.metric("Non-normal beats", f"{(result.predicted_class != 'Normal').mean() * 100:.1f}%")

# ---------------- Plot ----------------
st.subheader("ECG with beat classification")
win = st.slider("Window start (s)", 0.0, max(0.0, duration - 10), 0.0, 1.0)
t = np.arange(len(filtered)) / MODEL_FS
m = (t >= win) & (t <= win + 10)
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(t[m], filtered[m], lw=0.8, color="black")
for cls, color in COLORS.items():
    sel = (result.predicted_class == cls) & (result.time_s >= win) & (result.time_s <= win + 10)
    if sel.any():
        idx = (result.time_s[sel] * MODEL_FS).astype(int)
        ax.scatter(result.time_s[sel], filtered[idx], color=color, s=40, label=cls, zorder=3)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Filtered amplitude"); ax.grid(alpha=0.3); ax.legend(loc="upper right")
st.pyplot(fig)

col_a, col_b = st.columns([1, 2])
with col_a:
    st.subheader("Class counts")
    st.bar_chart(counts)
with col_b:
    st.subheader("Beat-level results")
    st.dataframe(result, height=260, use_container_width=True)
    st.download_button("Download results (CSV)", result.to_csv(index=False), "ecg_predictions.csv")

with st.expander("About this model"):
    st.write(
        f"Random Forest trained on MIT-BIH records {artifact.get('training_records')}. "
        "Classes: Normal / Supraventricular / Ventricular (simplified mapping). "
        "Beats are located with a simple peak detector, so results can differ from the "
        "annotation-based evaluation in the notebook. Confidence is the model's class "
        "probability, not a clinical risk score."
    )
