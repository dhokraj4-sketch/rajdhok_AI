from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

from src.preprocess import prepare_uploaded_image


MODEL_PATH = Path("models/mnist_model.keras")


st.set_page_config(page_title="MNIST Digit Classifier", layout="centered")

st.title("MNIST Handwritten Digit Classifier")
st.write("Upload a handwritten digit image and the trained neural network will predict the digit.")

if not MODEL_PATH.exists():
    st.warning("Model file not found. Please train the model first by running: python -m src.train")
    st.stop()

model = keras.models.load_model(MODEL_PATH)

uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=220)

    processed_image = prepare_uploaded_image(image)
    prediction = model.predict(processed_image, verbose=0)[0]
    predicted_digit = int(np.argmax(prediction))

    st.subheader(f"Predicted Digit: {predicted_digit}")
    st.write(f"Confidence: {prediction[predicted_digit] * 100:.2f}%")

    st.bar_chart({str(i): float(prediction[i]) for i in range(10)})
