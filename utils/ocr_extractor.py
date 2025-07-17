import cv2
import easyocr
import re


class OCRExtractor:
    def __init__(self, languages=None):
        # Initialize the EasyOCR reader with the given languages (default English)
        self.reader = easyocr.Reader(languages or ['en'])

    def extract_text(self, image, return_boxes=False):
        # Step 1: Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 2: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Step 3: Apply adaptive thresholding (use inverse binary)
        thresh = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Step 4: Convert to RGB (required by EasyOCR)
        thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

        # Step 5: Run OCR
        results = self.reader.readtext(thresh_rgb)

        # Step 6: Collect text, confidence, and bounding box
        extracted = []
        for (bbox, text, confidence) in results:
            extracted.append((text.strip(), confidence, bbox))

        if return_boxes:
            return extracted
        else:
            return " ".join([t[0] for t in extracted])

    def filter_serial_numbers(self, detections, min_conf=0.3):
        """Filter OCR results for possible seal/serial numbers"""
        serials = []
        for text, conf, _ in detections:
            cleaned = text.replace(" ", "")
            if conf > min_conf and re.match(r"^[A-Z]?\d{5,7}$", cleaned):
                serials.append((cleaned, conf))
        return serials


# Example usage
if __name__ == "__main__":
    import sys

    image_path = "seal_image.png"  # Replace with your file path
    image = cv2.imread(image_path)

    ocr = OCRExtractor()
    raw_results = ocr.extract_text(image, return_boxes=True)
    filtered = ocr.filter_serial_numbers(raw_results)

    print("Raw OCR Results:")
    for text, conf, _ in raw_results:
        print(f"Text: {text} | Confidence: {conf:.2f}")

    print("\nFiltered Seal/Serial Numbers:")
    for text, conf in filtered:
        print(f" Serial: {text} | Confidence: {conf:.2f}")
