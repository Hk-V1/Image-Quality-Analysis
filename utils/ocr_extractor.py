import cv2
import pytesseract

class OCRExtractor:
    def __init__(self, tesseract_cmd=None):
        """
        Optionally specify path to tesseract executable
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def extract_text(self, image):
        """
        Extracts text from the image using OCR
        :param image: Input image (BGR)
        :return: Extracted text (string)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 3)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Use a config to improve digit/ID accuracy (can tweak based on format)
        config = "--psm 6"  # Assume a block of text
        text = pytesseract.image_to_string(thresh, config=config)
        return text.strip()
