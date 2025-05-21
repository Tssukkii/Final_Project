import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import json

# Set page config
st.set_page_config(page_title="Weather Classifier", layout="centered")
st.title("🌤️ Weather Image Classifier")
st.write("Upload a weather image to classify it")

# === Load model and labels ===
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("weather_model.h5")

@st.cache_resource
def load_labels():
    with open("labels.json", "r") as f:
        return json.load(f)

model = load_model()
label_map = load_labels()

# === Image preprocessing ===
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image).astype(np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

# === Prediction ===
def predict(image):
    input_data = preprocess_image(image)
    predictions = model.predict(input_data)
    top_index = np.argmax(predictions[0])
    label = label_map[str(top_index)]
    confidence = predictions[0][top_index]
    return label, confidence

# === File uploader ===
uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Classify"):
        label, confidence = predict(image)
        st.success(f"🌦️ Predicted: **{label}** ({confidence * 100:.2f}% confidence)")
