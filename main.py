import streamlit as st
import cv2
import numpy as np
from PIL import Image
from image_utils.blur_detector import BlurDetector

st.title("Image Blur Detection and Quality Assessment")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    detector = BlurDetector()

    lap_var, is_sharp = detector.detect_blur(image_bgr)
    st.write(f"Laplacian Variance: {lap_var:.2f}")
    st.write(f"Sharp? {'Yes' if is_sharp else 'No'}")

    quality_score = detector.assess_quality(image_bgr)
    st.write(f"Quality Score: {quality_score:.2f}")

    fft_score, is_sharp_fft = detector.detect_blur_fft(image_bgr)
    st.write(f"FFT Blur Score: {fft_score:.4f}")
    st.write(f"Sharp (FFT)? {'Yes' if is_sharp_fft else 'No'}")
