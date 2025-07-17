import cv2
import easyocr

class OCRExtractor:
    def __init__(self, languages=None):
        self.reader = easyocr.Reader(languages or ['en'])

    def extract_text(self, image, return_boxes=False):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 3)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

        results = self.reader.readtext(rgb)

        extracted = []
        for (bbox, text, confidence) in results:
            extracted.append((text.strip(), confidence, bbox))

        if return_boxes:
            return extracted
        else:
            return " ".join([t[0] for t in extracted])
