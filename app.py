import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import os
from PIL import Image

# === Load model ===
model = load_model('model/weather_model.h5')
class_names = ['cloudy', 'rain', 'shine', 'sunrise']  # update based on your dataset

# === Streamlit App ===
st.set_page_config(page_title="Weather Classifier", layout="centered")
st.title("🌦️ Weather Image Classifier")
st.write("Upload a weather image and the model will classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array)

    # Predict
    prediction = model.predict(img_preprocessed)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Prediction: **{predicted_class}**")
