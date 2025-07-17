import cv2
import easyocr
import numpy as np
import re


class OCRExtractor:
    def __init__(self, languages=None):
        self.reader = easyocr.Reader(languages or ['en'])

    def preprocess_image(self, image):
        # Resize for better OCR
        image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

        # Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=30)

        # Sharpen
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharp = cv2.filter2D(denoised, -1, kernel)

        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(sharp, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        return rgb

    def extract_text(self, image, return_boxes=False):
        processed = self.preprocess_image(image)
        results = self.reader.readtext(processed)

        extracted = []
        for (bbox, text, confidence) in results:
            extracted.append((text.strip(), confidence, bbox))

        if return_boxes:
            return extracted
        else:
            return " ".join([t[0] for t in extracted])

    def filter_serial_numbers(self, detections, min_conf=0.3):
        serials = []
        for text, conf, _ in detections:
            cleaned = text.replace(" ", "").upper()
            if conf > min_conf and re.match(r"^[A-Z]{0,2}\d{4,7}$", cleaned):
                serials.append((cleaned, conf))
        return serials


if __name__ == "__main__":
    image_path = "seal_image.png"
    image = cv2.imread(image_path)

    ocr = OCRExtractor()
    raw_results = ocr.extract_text(image, return_boxes=True)
    filtered = ocr.filter_serial_numbers(raw_results)

    print("\nRaw OCR Results:")
    for text, conf, _ in raw_results:
        print(f"Text: {text} | Confidence: {conf:.2f}")

    print("\nFiltered Seal/Serial Numbers:")
    for text, conf in filtered:
        print(f" Serial: {text} | Confidence: {conf:.2f}")
