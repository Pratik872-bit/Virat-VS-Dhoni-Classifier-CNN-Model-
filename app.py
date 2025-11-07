import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ================================================================
# STEP 1: Load Model
# ================================================================
@st.cache_resource
def load_trained_model():
    model = load_model('dhoni_vs_kohli_final_model.h5')
    return model

model = load_trained_model()

# ================================================================
# STEP 2: App UI
# ================================================================
st.set_page_config(page_title="Dhoni vs Kohli Classifier 🏏", layout="centered")
st.title("🏏 Dhoni vs Kohli Image Classifier")
st.markdown("Upload an image, and the model will tell whether it's **MS Dhoni**, **Virat Kohli**, or someone **else**,the model accuracy is 85% so some time it make mistake👇")

uploaded_file = st.file_uploader("📸 Choose an image...", type=["jpg", "jpeg", "png"])

# ================================================================
# STEP 3: Prediction Logic
# ================================================================
if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("🔍 Analyzing...")

    # Preprocess
    img = img.resize((150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    # Predict
    pred = model.predict(x)[0][0]

    # Define confidence thresholds
    dhoni_conf = 1 - pred
    kohli_conf = pred

    st.write(f"Raw Prediction Value: {pred:.4f}")

    # Decision Logic
    if kohli_conf > 0.75:
        st.success(f"🏏 Predicted: **Virat Kohli** (Confidence: {kohli_conf:.2f})")
        st.progress(float(kohli_conf))
    elif dhoni_conf > 0.75:
        st.success(f"🏏 Predicted: **MS Dhoni** (Confidence: {dhoni_conf:.2f})")
        st.progress(float(dhoni_conf))
    else:
        st.warning("🤔 The model is **not confident enough**. This might be **someone else (Unknown Person)**.")
        st.progress(0.5)

else:
    st.info("Please upload an image to get started.")
