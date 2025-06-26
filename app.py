import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image

# ✅ MUST be the first Streamlit command
st.set_page_config(page_title="Rice Plant Disease Detector", layout="centered")

# Load model and class names only once
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('rice_disease_model.h5')
    with open('class_names.pkl', 'rb') as f:
        class_names = pickle.load(f)
    return model, class_names

model, class_names = load_model()

# Preprocess uploaded image
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = img_array / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)

# UI
st.title("🌾 Rice Plant Disease Predictor")
st.markdown("Upload a rice leaf image to classify the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Leaf Image', use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            input_tensor = preprocess_image(image)
            prediction = model.predict(input_tensor)[0]
            predicted_index = np.argmax(prediction)
            confidence = prediction[predicted_index] * 100
            predicted_class = class_names[predicted_index]

            st.success(f"✅ **Prediction:** {predicted_class}")
            st.info(f"📊 Confidence: {confidence:.2f}%")
