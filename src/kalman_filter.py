import cv2
import numpy as np


class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)

        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.kf.statePost = np.zeros(4, dtype=np.float32)

        self.color_histogram = None

    def predict(self):
        return self.kf.predict()

    def update(self, measurement):
        return self.kf.correct(measurement)

    def set_color_histogram(self, frame: np.ndarray, bounding_box: tuple) -> None:
        """Compute a color histogram for a given region of interest (bounding box)
        from a frame and store it as the object's color histogram.

        Args:
            frame (ndarray): The frame to compute the color histogram from.
            bounding_box (tuple): The bounding box of the object in the frame.
        """
        x1, y1, x2, y2 = bounding_box
        roi = frame[y1:y2, x1:x2]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        self.color_histogram = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(self.color_histogram, self.color_histogram, 0, 255, cv2.NORM_MINMAX)

    def compare_color_histogram(self, frame: np.ndarray, bounding_box: tuple) -> float:
        """Compare a given region of interest (bounding box) from a frame
        with the stored color histogram of the object.

        Args:
            frame (ndarray): The frame to compare with the stored color histogram.
            bounding_box (tuple): The bounding box of the object in the frame.

        Returns:
            float: The Bhattacharyya distance between the two histograms.
        """
        x1, y1, x2, y2 = bounding_box
        roi = frame[y1:y2, x1:x2]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        similarity = cv2.compareHist(self.color_histogram, hist, cv2.HISTCMP_BHATTACHARYYA)
        return similarity
