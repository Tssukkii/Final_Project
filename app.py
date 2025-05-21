import streamlit as st
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

st.title("🌤️ Weather Image Classifier")
st.write("Upload an image to predict the weather condition.")

# Load TFLite model
interpreter = tflite.Interpreter(model_path="weather_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    classes = ['cloudy', 'rain', 'shine', 'sunrise']  # Adjust as needed
    pred_class = classes[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.success(f"**Prediction:** {pred_class} ({confidence:.2f}%)")
