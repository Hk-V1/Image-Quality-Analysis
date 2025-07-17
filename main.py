import cv2
import streamlit as st
from utils.blur_detector import BlurDetector
from utils.ocr_utils import OCRUtils
from PIL import Image
import numpy as np

class SerialNumberApp:
    def __init__(self):
        self.blur_detector = BlurDetector()
        self.ocr_utils = OCRUtils()
        
    def run(self):
        st.set_page_config(page_title="Serial Number Detector", layout="wide")
        st.title("📷 Real-time Serial Number Detection")
        
        # Sidebar settings
        st.sidebar.header("Settings")
        blur_threshold = st.sidebar.slider("Blur Threshold", 50, 200, 100)
        quality_threshold = st.sidebar.slider("Quality Threshold", 20, 100, 50)
        
        # OCR settings
        st.sidebar.subheader("OCR Settings")
        enable_preprocessing = st.sidebar.checkbox("Enhanced Preprocessing", True)
        upscale_image = st.sidebar.checkbox("Upscale Small Images", True)
        
        # Update thresholds
        self.blur_detector.set_threshold(blur_threshold)
        
        # Camera input
        camera_input = st.camera_input("Take a picture")
        
        if camera_input:
            # Convert to OpenCV format
            image = Image.open(camera_input)
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Create columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
                
            with col2:
                st.subheader("Analysis Results")
                
                # Blur detection
                blur_score, is_sharp = self.blur_detector.detect_blur(opencv_image)
                st.metric("Blur Score", f"{blur_score:.2f}", 
                         "Sharp ✅" if is_sharp else "Blurry ❌")
                
                # Quality assessment
                quality_score = self.blur_detector.assess_quality(opencv_image)
                st.metric("Quality Score", f"{quality_score:.2f}")
                
                # OCR extraction
                if is_sharp:
                    with st.spinner("Extracting serial number..."):
                        serial_number = self.ocr_utils.extract_serial_number(opencv_image)
                        
                    if serial_number:
                        st.success(f"🎯 Serial Number: **{serial_number}**")
                        
                        # Show confidence and details
                        if len(serial_number) == 6 and serial_number.isdigit():
                            st.info("✅ Pattern: 6-digit component number")
                        elif serial_number.isdigit():
                            st.info(f"✅ Pattern: {len(serial_number)}-digit number")
                        else:
                            st.info(f"✅ Pattern: Alphanumeric ({len(serial_number)} chars)")
                            
                    else:
                        st.warning("⚠️ No serial number detected")
                        st.info("💡 Try: Better lighting, closer shot, or different angle")
                else:
                    st.error("❌ Image too blurry for OCR")
                    st.info("💡 Hold camera steady and ensure good lighting")

if __name__ == "__main__":
    app = SerialNumberApp()
    app.run()
