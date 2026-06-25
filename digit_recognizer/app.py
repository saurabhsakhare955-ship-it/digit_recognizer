import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import os

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="🧠",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1f77b4; text-align: center; }
    .subtitle   { font-size: 1rem; color: #555; text-align: center; margin-bottom: 1.5rem; }
    .result-box { padding: 1.2rem; border-radius: 10px; text-align: center; font-size: 1.4rem; font-weight: bold; }
    .spam-box   { background: #d4edda; color: #155724; }
    .ham-box    { background: #f8d7da; color: #721c24; }
    .metric-card { background: #f0f2f6; border-radius: 8px; padding: 1rem; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🧠 Handwritten Digit Recognizer</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">CNN trained on MNIST Dataset | ~99% Accuracy</p>', unsafe_allow_html=True)

# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists("model/digit_model.h5"):
        st.error("❌ Model not found! Please run `python train_model.py` first.")
        st.stop()
    return tf.keras.models.load_model("model/digit_model.h5")

model = load_model()
st.success("✅ Model loaded successfully!")

# ── Sidebar Info ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Model Info")
    st.markdown("""
    **Architecture:** CNN (Convolutional Neural Network)

    **Dataset:** MNIST
    - 60,000 training images
    - 10,000 test images

    **Layers:**
    - Conv2D (32 filters)
    - Conv2D (64 filters)
    - Conv2D (64 filters)
    - Dense (128 units)
    - Dropout (0.3)
    - Output (10 classes)

    **Optimizer:** Adam
    **Loss:** Categorical Crossentropy
    **Accuracy:** ~99%
    """)

    if os.path.exists("model/training_plot.png"):
        st.image("model/training_plot.png", caption="Training History")

# ── Main App ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📤 Upload a Handwritten Digit Image")
st.info("💡 Tip: Upload a clear image of a single digit (0-9). Works best with dark digit on white background.")

uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["png", "jpg", "jpeg"],
    help="Upload a handwritten digit image"
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Original Image:**")
        st.image(uploaded_file, width=200)

    # Preprocess image
    image = Image.open(uploaded_file).convert("L")   # Grayscale
    image = image.resize((28, 28), Image.LANCZOS)

    # Auto-detect if we need to invert (MNIST: white digit on black bg)
    img_array_raw = np.array(image)
    if img_array_raw.mean() > 127:
        image = ImageOps.invert(image)               # Invert if bg is white

    img_array = np.array(image).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    with col2:
        st.markdown("**Processed (28×28):**")
        st.image(image, width=200, caption="Input to CNN")

    # ── Prediction ─────────────────────────────────────────────────────────────
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_digit = int(np.argmax(predictions))
    confidence = float(predictions[predicted_digit]) * 100
    second_best = int(np.argsort(predictions)[-2])
    second_conf = float(predictions[second_best]) * 100

    st.markdown("---")
    st.subheader("🎯 Prediction Result")

    c1, c2, c3 = st.columns(3)
    c1.metric("🔢 Predicted Digit", predicted_digit)
    c2.metric("📈 Confidence", f"{confidence:.1f}%")
    c3.metric("2nd Guess", f"{second_best} ({second_conf:.1f}%)")

    # Confidence bar
    if confidence >= 90:
        st.success(f"✅ High confidence! The digit is **{predicted_digit}** ({confidence:.1f}%)")
    elif confidence >= 70:
        st.warning(f"⚠️ Moderate confidence. The digit is likely **{predicted_digit}** ({confidence:.1f}%)")
    else:
        st.error(f"❌ Low confidence ({confidence:.1f}%). Try a clearer image.")

    # ── All Probabilities ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Probability Distribution (All Digits)")

    import pandas as pd
    prob_df = pd.DataFrame({
        "Digit": [str(i) for i in range(10)],
        "Probability (%)": [round(float(p) * 100, 2) for p in predictions]
    })

    # Highlight predicted digit
    def highlight_max(row):
        return ['background-color: #1f77b4; color: white; font-weight: bold'
                if row["Digit"] == str(predicted_digit) else '' for _ in row]

    st.dataframe(
        prob_df.style.apply(highlight_max, axis=1),
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        pd.DataFrame({"Probability": [float(p) for p in predictions]},
                     index=[str(i) for i in range(10)])
    )

else:
    # Show sample predictions if available
    if os.path.exists("model/sample_predictions.png"):
        st.markdown("---")
        st.subheader("🖼️ Sample Predictions from Test Set")
        st.image("model/sample_predictions.png", use_column_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.85rem;'>
    Built with TensorFlow & Streamlit | Codec Technologies AI Internship Project
</div>
""", unsafe_allow_html=True)
