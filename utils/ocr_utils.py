import cv2
import numpy as np
import pytesseract
import re

class OCRUtils:
    def __init__(self):
        # Configure Tesseract path if needed (Windows)
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        pass
    
    def preprocess_image(self, image):
        """
        Preprocess image for better OCR results on small components
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize image for better OCR (upscale small text)
        height, width = gray.shape
        if height < 100 or width < 100:
            scale_factor = 3
            gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        # Sharpen for embossed text
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # Adaptive threshold for varying lighting
        adaptive_thresh = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations to clean up
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def extract_serial_number(self, image):
        """
        Extract serial number from image using OCR - optimized for small components
        """
        processed = self.preprocess_image(image)
        
        # Multiple OCR configurations for different scenarios
        configs = [
            '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789',  # Single word, digits only
            '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',  # Single text line
            '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',  # Single uniform block
            '--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789',  # Raw line, digits only
        ]
        
        for config in configs:
            try:
                # Extract text
                text = pytesseract.image_to_string(processed, config=config)
                
                # Clean extracted text
                text = text.strip().replace('\n', ' ').replace('\r', ' ')
                
                # Find serial number using patterns
                serial_number = self.find_serial_patterns(text)
                
                if serial_number:
                    return serial_number
                    
            except Exception as e:
                continue
        
        # If no config worked, try with original image
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray, config='--oem 3 --psm 8')
            cleaned = re.sub(r'[^0-9A-Z]', '', text.upper())
            if len(cleaned) >= 4 and cleaned.isalnum():
                return cleaned
        except:
            pass
            
        return None
    
    def find_serial_patterns(self, text):
        """
        Find serial number patterns in extracted text - optimized for component numbers
        """
        # Clean the text first
        cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Patterns for component serial numbers (like 158105)
        patterns = [
            r'(\d{6})',                                      # 6 digits (like 158105)
            r'(\d{4,8})',                                    # 4-8 digits
            r'([A-Z]{1,2}\d{4,6})',                         # 1-2 letters + 4-6 digits
            r'(\d{3,4}[A-Z]{1,2}\d{2,4})',                 # Mixed pattern
            r'(?:S/N|SN|SERIAL)[:\s]*([A-Z0-9]{4,10})',     # With prefix
            r'([A-Z0-9]{4,10})',                            # General alphanumeric
        ]
        
        # Try patterns on cleaned text
        for pattern in patterns:
            matches = re.findall(pattern, cleaned_text)
            if matches:
                match = matches[0]
                if isinstance(match, tuple):
                    match = match[0]
                
                # Validate match
                if len(match) >= 4 and self.is_valid_serial(match):
                    return match
        
        # If no pattern matches but we have clean digits/letters
        if len(cleaned_text) >= 4 and cleaned_text.isalnum():
            return cleaned_text
        
        # Try on original text with looser cleaning
        loose_clean = re.sub(r'[^A-Z0-9\s]', '', text.upper())
        numbers = re.findall(r'\d{4,8}', loose_clean)
        if numbers:
            return numbers[0]
        
        return None
    
    def is_valid_serial(self, serial):
        """
        Validate if extracted text looks like a valid serial number
        """
        # Should be reasonable length (4-12 for component numbers)
        valid_length = 4 <= len(serial) <= 12
        
        # Should not be all same character
        not_repetitive = len(set(serial)) > 1
        
        # Should have at least some digits
        has_digit = any(c.isdigit() for c in serial)
        
        # Should not have common OCR errors
        no_common_errors = not any(char in serial for char in ['I', 'l', '|'])
        
        return valid_length and not_repetitive and has_digit
    
    def extract_with_boxes(self, image):
        """
        Extract text with bounding boxes for debugging
        """
        processed = self.preprocess_image(image)
        
        # Get detailed OCR data
        data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
        
        results = []
        for i, text in enumerate(data['text']):
            if text.strip():
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                conf = data['conf'][i]
                results.append({
                    'text': text,
                    'bbox': (x, y, w, h),
                    'confidence': conf
                })
        
        return results
