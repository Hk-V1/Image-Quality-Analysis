import cv2
import easyocr

class OCRExtractor:
    def __init__(self, languages=None):
        """
        Initializes the EasyOCR reader.
        :param languages: List of language codes (e.g., ['en'])
        """
        self.reader = easyocr.Reader(languages or ['en'])

    def extract_text(self, image):
        """
        Extracts text from the image using EasyOCR
        :param image: Input image (BGR)
        :return: Extracted text (string)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 3)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # EasyOCR works with RGB format
        rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        results = self.reader.readtext(rgb)

        # Concatenate results
        text = " ".join([res[1] for res in results])
        return text.strip()
