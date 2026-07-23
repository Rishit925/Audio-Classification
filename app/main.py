import streamlit as st
import time
from app.predict import predict
import tempfile
import os

st.set_page_config(
    page_title="Quran Reciter Classification",
    page_icon="🎧",
    layout="wide"
)

# =====================================
# Sidebar
# =====================================

with st.sidebar:

    st.title("📖 About")

    st.write(
        """
This application classifies **Quran Reciters**
from uploaded audio recordings using a
Convolutional Neural Network trained on
Mel Spectrograms.
"""
    )

    st.divider()

    with st.expander("🧠 Model Information", expanded=True):

        st.markdown("""
- **Architecture:** CNN
- **Framework:** PyTorch
- **Classes:** 12 Reciters
- **Features:** Mel Spectrogram
- **Regularization:** BatchNorm + Dropout
""")

    with st.expander("⚙ Audio Configuration"):

        st.markdown("""
- Sample Rate : **22050 Hz**
- Duration : **5 Seconds**
- Mel Bands : **128**
- Input Size : **128 × 256**
""")

    with st.expander("📁 Supported Format"):

        st.success("WAV (.wav)")

    st.divider()

    st.caption("Developed by")
    st.markdown("**Rishit Mahindru**")

# =====================================
# Header
# =====================================

st.title("🎧 Quran Reciter Classification")

st.markdown(
"""
Deep Learning based Audio Classification using
**Convolutional Neural Networks (CNN)** and
**Mel Spectrograms**.
"""
)

st.divider()

# =====================================
# Upload
# =====================================

uploaded_file = st.file_uploader(
    "Upload Quran Audio",
    type=["wav"]
)

if uploaded_file:

    left, right = st.columns([1, 1])

    # -----------------------------
    # LEFT PANEL
    # -----------------------------

    with left:

        st.subheader("🎵 Uploaded Audio")

        st.audio(uploaded_file)

        st.info(f"📄 {uploaded_file.name}")

    # -----------------------------
    # RIGHT PANEL
    # -----------------------------

    with right:

        st.subheader("Prediction")

        if st.button(
            "Predict Reciter",
            use_container_width=True
        ):

            start = time.time()

            with st.spinner("Generating Mel Spectrogram..."):

                time.sleep(0.3)

            with st.spinner("Running CNN Inference..."):

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_audio_path = tmp_file.name

                result = predict(temp_audio_path)

                os.remove(temp_audio_path)

            end = time.time()

            confidence = result["confidence"]

            reciter = result["reciter"].replace("_", " ")

            st.success("✅ Prediction Completed")

            st.metric(
                    "Predicted Reciter",
                    reciter
                )

            st.progress(confidence / 100)

            if confidence >= 90:
                confidence_level = "🟢 High"
                message = "The model is highly confident in this prediction."

            elif confidence >= 70:
                confidence_level = "🟡 Moderate"
                message = "The prediction is reasonably confident."

            else:
                confidence_level = "🔴 Low"
                message = (
                        "The model has lower confidence. "
                        "Consider verifying with another audio sample."
                    )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                        "Prediction Confidence",
                        confidence_level
                    )

            with col2:

                st.metric(
                        "Processing Time",
                        f"{(end-start)*1000:.0f} ms"
                    )

            st.caption(message)

            

st.divider()

st.caption(
    "Built using PyTorch • Hugging Face Hub • Streamlit"
)