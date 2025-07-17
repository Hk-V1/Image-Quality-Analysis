import cv2
import numpy as np

class BlurDetector:
    def __init__(self, threshold=100):
        self.threshold = threshold

    def set_threshold(self, threshold):
        self.threshold = threshold

    def detect_blur(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_sharp = laplacian_var > self.threshold
        return laplacian_var, is_sharp

    def assess_quality(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = gray.std()
        brightness = gray.mean()
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
        quality_score = (contrast + edge_density * 1000) / 2
        return quality_score

    def detect_blur_fft(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        h, w = gray.shape
        center_h, center_w = h // 2, w // 2
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (center_w, center_h), min(center_h, center_w) // 3, 255, -1)
        mask = 255 - mask
        high_freq_energy = np.sum(magnitude * mask)
        total_energy = np.sum(magnitude)
        blur_score = high_freq_energy / total_energy
        return blur_score, blur_score > 0.1
