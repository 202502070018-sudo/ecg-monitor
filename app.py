import io
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ecg_utils import (
    MODEL_FS,
    bandpass_filter,
    detect_r_peaks,
    segment_beats,
    to_model_rate,
)

MODEL_PATH = "ecg_random_forest_baseline.joblib"

COLORS = {
    "Normal": "tab:green",
    "Supraventricular": "tab:orange",
    "Ventricular": "tab:red",
}

st.set_page_config(
    page_title="AI ECG Monitor",
    page_icon="🫀",
    layout="wide"
)

st.title("🫀 AI-Based ECG Monitoring & Abnormality Detection")

st.warning(
    "Research/education demo only. NOT a medical device. "
    "Do not use for diagnosis or treatment decisions."
)


# ---------------- Load trained model ----------------

@st.cache_resource
def load_artifact():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner="Downloading MIT-BIH record...")
def load_mitbih(record_id):
    import wfdb

    rec = wfdb.rdrecord(
        record_id,
        pn_dir="mitdb",
        sampto=360 * 60
    )

    return rec.p_signal[:, 0], rec.fs


artifact = load_artifact()

if artifact is None:
    st.error(
        f"{MODEL_PATH} not found. "
        "Place the trained model file next to app.py."
    )
    st.stop()

model = artifact["model"]

pre_s = artifact["pre_seconds"]
post_s = artifact["post_seconds"]


# ---------------- Sidebar: ECG input ----------------

st.sidebar.header("ECG input")

source = st.sidebar.radio(
    "Source",
    ["MIT-BIH demo record", "Upload CSV"]
)

signal = None
fs = None

if source == "MIT-BIH demo record":

    rid = st.sidebar.selectbox(
        "Record",
        ["100", "119", "200", "208", "233"],
        index=2
    )

    try:
        signal, fs = load_mitbih(rid)

    except Exception as e:
        st.error(f"Could not download MIT-BIH record: {e}")

else:

    up = st.sidebar.file_uploader(
        "CSV with an ECG column",
        type=["csv", "txt"]
    )

    fs = st.sidebar.number_input(
        "Sampling rate (Hz)",
        min_value=50,
        max_value=2000,
        value=360
    )

    if up is not None:

        try:
            df = pd.read_csv(io.BytesIO(up.getvalue()))

            # Find numerical ECG columns
            num_cols = df.select_dtypes(
                include="number"
            ).columns.tolist()

            # Handle headerless numeric CSV
            if not num_cols:
                df = pd.read_csv(
                    io.BytesIO(up.getvalue()),
                    header=None
                )

                num_cols = df.select_dtypes(
                    include="number"
                ).columns.tolist()

            if not num_cols:
                st.error("No numerical ECG column found.")

            else:

                col = st.sidebar.selectbox(
                    "ECG column",
                    num_cols
                )

                signal = pd.to_numeric(
                    df[col],
                    errors="coerce"
                ).dropna().to_numpy(dtype=float)

        except Exception as e:
            st.error(f"CSV reading error: {e}")


# ---------------- Validate input ----------------

if signal is None:
    st.info("Choose a demo record or upload a CSV to begin.")
    st.stop()

if len(signal) < 2:
    st.error("ECG signal is too short.")
    st.stop()

if not np.all(np.isfinite(signal)):
    st.error("ECG signal contains invalid values.")
    st.stop()

if fs is None or fs <= 0:
    st.error("Invalid sampling rate.")
    st.stop()


# ---------------- Signal processing ----------------

try:

    sig = to_model_rate(signal, fs)

    filtered = bandpass_filter(
        sig,
        MODEL_FS
    )

    peaks = detect_r_peaks(filtered)

    X, kept = segment_beats(
        filtered,
        peaks,
        pre_s,
        post_s
    )

except Exception as e:
    st.error(f"Signal processing error: {e}")
    st.stop()


if len(X) == 0:
    st.error(
        "No beats detected. Check the sampling rate "
        "and confirm that the uploaded signal is a raw ECG."
    )
    st.stop()


# ---------------- Model prediction ----------------

try:

    pred = model.predict(X)

    proba = model.predict_proba(X).max(axis=1)

except Exception as e:
    st.error(f"Model prediction error: {e}")
    st.stop()


result = pd.DataFrame({
    "time_s": kept / MODEL_FS,
    "predicted_class": pred,
    "confidence": proba.round(3)
})


# ---------------- Summary metrics ----------------

duration = len(filtered) / MODEL_FS

hr = (
    60 * (len(kept) - 1) /
    ((kept[-1] - kept[0]) / MODEL_FS)
    if len(kept) > 1 and kept[-1] > kept[0]
    else 0
)

counts = result["predicted_class"].value_counts()

st.subheader("ECG Analysis Summary")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Duration", f"{duration:.1f} s")
c2.metric("Beats detected", len(kept))
c3.metric("Average heart rate", f"{hr:.0f} bpm")

non_normal = (
    result["predicted_class"] != "Normal"
).mean() * 100

c4.metric("Non-normal beats", f"{non_normal:.1f}%")


# ---------------- ECG visualization ----------------

st.subheader("ECG with beat classification")

window_seconds = min(10.0, duration)

if duration > window_seconds:

    max_start = duration - window_seconds

    win = st.slider(
        "Window start (s)",
        min_value=0.0,
        max_value=float(max_start),
        value=0.0,
        step=min(1.0, float(max_start))
    )

else:

    win = 0.0

    st.info(
        f"Recording duration is {duration:.1f} seconds. "
        "Displaying the complete ECG signal."
    )


t = np.arange(len(filtered)) / MODEL_FS

end_time = min(
    win + window_seconds,
    duration
)

m = (t >= win) & (t <= end_time)

fig, ax = plt.subplots(figsize=(14, 4))

ax.plot(
    t[m],
    filtered[m],
    lw=0.8,
    color="black",
    label="ECG signal"
)

# Mark classified R-peaks
for cls, color in COLORS.items():

    sel = (
        (result.predicted_class == cls) &
        (result.time_s >= win) &
        (result.time_s <= end_time)
    )

    if sel.any():

        idx = (
            result.time_s[sel] * MODEL_FS
        ).astype(int)

        ax.scatter(
            result.time_s[sel],
            filtered[idx],
            color=color,
            s=35,
            label=cls,
            zorder=3
        )

ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Filtered amplitude")
ax.set_title("ECG Beat Classification")

ax.grid(alpha=0.3)
ax.legend(loc="upper right")

st.pyplot(fig)
plt.close(fig)


# ---------------- Class distribution ----------------

col_a, col_b = st.columns([1, 2])

with col_a:

    st.subheader("Class counts")

    st.bar_chart(counts)


with col_b:

    st.subheader("Beat-level results")

    st.dataframe(
        result,
        height=260,
        use_container_width=True
    )

    st.download_button(
        "Download results (CSV)",
        result.to_csv(index=False),
        "ecg_predictions.csv",
        "text/csv"
    )


# ---------------- Model information ----------------

with st.expander("About this model"):

    st.write(
        f"Random Forest trained on MIT-BIH records "
        f"{artifact.get('training_records')}."
    )

    st.write(
        "Classes: Normal / Supraventricular / "
        "Ventricular (simplified mapping)."
    )

    st.write(
        "Beats are located using a simple R-peak detector. "
        "Results may differ from annotation-based evaluation."
    )

    st.write(
        "Confidence represents the model's predicted class "
        "probability, not a clinical risk score."
    )

    st.write(
        "This application is intended for research and "
        "educational demonstration only."
    )
