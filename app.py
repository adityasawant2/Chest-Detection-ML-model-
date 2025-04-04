import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from preprocess import preprocess_uploaded_image
from gradcam import get_gradcam_heatmap, overlay_heatmap

# Load the model
model = tf.keras.models.load_model("models/mobilenet_pneumonia.keras")  # Updated extension
# Sort class names alphabetically for consistency
class_names = sorted(["NORMAL", "PNEUMONIA", "COVID19", "TUBERCULOSIS"])  # e.g., ["COVID19", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"]

st.title("AI-Powered Disease Diagnosis from X-rays")
st.write("Upload an X-ray image to classify it as Normal, Pneumonia, COVID-19, or Tuberculosis.")

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess the image
    img = preprocess_uploaded_image(uploaded_file)
    img_array = img[np.newaxis, ...]

    # Make prediction
    prediction = model.predict(img_array)
    pred_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Display the image and prediction
    st.image(uploaded_file, caption="Uploaded X-ray", use_container_width=True)
    st.write(f"Prediction: **{pred_class}** (Confidence: {confidence:.2f}%)")

    # Generate and display Grad-CAM heatmap
    heatmap = get_gradcam_heatmap(model, img_array, last_conv_layer_name="conv_pw_13")
    file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)  # Use getvalue() for streamlit
    img_for_heatmap = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_for_heatmap is None:
        st.error("Error loading image for heatmap processing.")
    else:
        img_for_heatmap = cv2.resize(img_for_heatmap, (224, 224))

        # Overlay the Grad-CAM heatmap
        superimposed_img = overlay_heatmap(heatmap, img_for_heatmap)
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        st.image(superimposed_img, caption="Grad-CAM Heatmap (Regions of Interest)", use_container_width=True)