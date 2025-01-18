import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self._initialize_matrices()
        self.prev_position = None
        self.velocity = 0.0
        self.color_histogram = None

    def _initialize_matrices(self):
        """Initialize the matrices for the Kalman Filter."""
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

        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-4
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.kf.statePost = np.zeros(4, dtype=np.float32)

    def predict(self):
        """Predict the next state using the Kalman Filter."""
        return self.kf.predict()

    def update(self, measurement):
        """Update the Kalman Filter with a new measurement."""
        corrected = self.kf.correct(measurement)
        if self.prev_position is not None:
            self.velocity = np.linalg.norm(corrected[:2] - self.prev_position)
        self.prev_position = corrected[:2].copy()
        return corrected

    def set_color_histogram(self, frame: np.ndarray, bounding_box: tuple) -> None:
        """Set the color histogram for a given bounding box in the frame."""
        x1, y1, x2, y2 = bounding_box
        roi = frame[y1:y2, x1:x2]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        self.color_histogram = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(self.color_histogram, self.color_histogram, 0, 255, cv2.NORM_MINMAX)

    def compare_color_histogram(self, frame: np.ndarray, bounding_box: tuple) -> float:
        """Compare the stored color histogram with the histogram of a given bounding box."""
        x1, y1, x2, y2 = bounding_box
        roi = frame[y1:y2, x1:x2]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        return cv2.compareHist(self.color_histogram, hist, cv2.HISTCMP_BHATTACHARYYA)


def create_cost_matrix(kf_objects, measurements, bounding_boxes, frame):
    """
    Create a cost matrix for the Hungarian algorithm based on position and color histogram.

    Args:
        kf_objects (list): List of KalmanFilter objects.
        measurements (list): List of measurement vectors.
        bounding_boxes (list): List of bounding boxes corresponding to measurements.
        frame (ndarray): The current frame.

    Returns:
        ndarray: The computed cost matrix.
    """
    if not kf_objects or not measurements:
        return None

    cost_matrix = np.array([
        [np.linalg.norm(kf.predict()[:2] - measurement) +
         (kf.compare_color_histogram(frame, bbox) if kf.color_histogram is not None else 0)
         for measurement, bbox in zip(measurements, bounding_boxes)]
        for kf in kf_objects
    ])
    return cost_matrix


def update_kalman_filters(kf_objects, measurements, bounding_boxes, frame, cost_matrix):
    """
    Update Kalman filter objects with new measurements and bounding boxes.

    Args:
        kf_objects (list): List of KalmanFilter objects.
        measurements (list): List of measurement vectors.
        bounding_boxes (list): List of bounding boxes corresponding to measurements.
        frame (ndarray): The current frame.
        cost_matrix (ndarray): The cost matrix for the Hungarian algorithm.

    """
    if cost_matrix is not None:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for r, c in zip(row_ind, col_ind):
            kf_objects[r].update(measurements[c])
            kf_objects[r].set_color_histogram(frame, bounding_boxes[c])

    while len(kf_objects) < len(measurements):
        new_kf = KalmanFilter()
        new_kf.set_color_histogram(frame, bounding_boxes[len(kf_objects)])
        kf_objects.append(new_kf)


def draw_predictions(frame: np.ndarray, kf_objects: list, ball_positions: dict) -> None:
    """
    Draw predictions and trails of tracked objects on the frame.

    Args:
        frame (ndarray): The frame on which to draw.
        kf_objects (list): List of KalmanFilter objects.
        ball_positions (dict): Dictionary of ball positions with their IDs.
    """
    colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0)]

    for i, kf in enumerate(kf_objects):
        predicted = kf.predict()
        px, py = int(predicted[0]), int(predicted[1])
        if px == 0 or py == 0:
            continue

        color = colors[i % len(colors)]
        cv2.circle(frame, (px, py), 10, color, -1)
        cv2.putText(frame, f"Ball {i + 1}", (px + 10, py + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"{kf.velocity:.2f} px/frame", (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if i not in ball_positions:
            ball_positions[i] = []
        ball_positions[i].append((px, py))

        overlay = frame.copy()
        for j in range(1, len(ball_positions[i])):
            cv2.line(overlay, ball_positions[i][j - 1], ball_positions[i][j], color, 2)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)