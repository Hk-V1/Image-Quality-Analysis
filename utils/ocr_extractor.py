import cv2
import easyocr

class OCRExtractor:
    def __init__(self, languages=None):
        self.reader = easyocr.Reader(languages or ['en'])

    def extract_text(self, image, return_boxes=False, enhance_contrast=True, scale=2.0):
        """
        Extract text using EasyOCR with optional enhancements.

        :param image: Input image (BGR format)
        :param return_boxes: Whether to return bounding boxes with results
        :param enhance_contrast: Use histogram equalization to improve readability
        :param scale: Resize factor to improve small text detection
        :return: Text string or list of (text, confidence, bbox)
        """
        # Resize image to improve OCR for small text
        if scale != 1.0:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale and enhance contrast if requested
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if enhance_contrast:
            gray = cv2.equalizeHist(gray)

        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # Read text using EasyOCR
        results = self.reader.readtext(rgb)

        extracted = [(text.strip(), confidence, bbox) for (bbox, text, confidence) in results]

        return extracted if return_boxes else " ".join([t[0] for t in extracted])
