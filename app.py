
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🔬",
    layout="centered"
)

IMG_SIZE = (128, 128)
MODEL_PATH = "best_skin_cancer_cnn.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

st.title("🔬 Skin Cancer Detection")
st.write("CNN Based Skin Image Classification")

try:
    cnn_model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a Skin Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Skin Image")

    if st.button("Predict"):

        img = image.resize(IMG_SIZE)

        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = cnn_model.predict(
            img_array,
            verbose=0
        )[0][0]

        st.subheader("Prediction Result")

        if prediction >= 0.5:
            st.error("Prediction: Malignant")
            confidence = prediction * 100
        else:
            st.success("Prediction: Benign")
            confidence = (1 - prediction) * 100

        st.write(f"Confidence: {confidence:.2f}%")

        st.warning(
            "This AI prediction is for educational purposes only. "
            "It cannot diagnose or rule out skin cancer. "
            "Please consult a qualified dermatologist."
        )
