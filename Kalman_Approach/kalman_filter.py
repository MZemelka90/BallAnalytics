import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


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

        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-4
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.kf.statePost = np.zeros(4, dtype=np.float32)

        self.prev_position = None
        self.velocity = 0.0

    def predict(self):
        return self.kf.predict()

    def update(self, measurement):
        corrected = self.kf.correct(measurement)
        if self.prev_position is not None:
            self.velocity = np.linalg.norm(corrected[:2] - self.prev_position)
        self.prev_position = corrected[:2].copy()
        return corrected

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


def create_cost_matrix(kf_objects, measurements, bounding_boxes, frame):
    """
    Create a cost matrix for the Hungarian algorithm to pair Kalman filter objects
    with measurements from a frame.

    Args:
        kf_objects (list): A list of KalmanFilter objects.
        measurements (list): A list of numpy arrays representing the positions of
            detected balls in the frame.
        bounding_boxes (list): A list of tuples representing the bounding boxes
            of the detected balls in the frame.
        frame (ndarray): The frame from which the measurements and bounding boxes
            were obtained.

    Returns:
        A 2D numpy array representing the cost matrix. If the input lists are empty,
        None is returned.
    """
    cost_matrix = []
    for kf in kf_objects:
        costs = []
        for i, bbox in enumerate(bounding_boxes):
            position = measurements[i]
            predicted = kf.predict()
            position_cost = np.linalg.norm(predicted[:2] - position)
            color_cost = kf.compare_color_histogram(frame, bbox) if kf.color_histogram is not None else 0
            total_cost = position_cost + color_cost

            costs.append(total_cost)
        cost_matrix.append(costs)
    return np.array(cost_matrix) if cost_matrix else None


def update_kalman_filters(kf_objects, measurements, bounding_boxes, frame, cost_matrix):
    """
    Update the Kalman filter objects given the measurements from a frame, bounding boxes,
    and the cost matrix computed for the Hungarian algorithm.

    Args:
        kf_objects (list): A list of KalmanFilter objects.
        measurements (list): A list of numpy arrays representing the positions of
            detected balls in the frame.
        bounding_boxes (list): A list of tuples representing the bounding boxes
            of the detected balls in the frame.
        frame (ndarray): The frame from which the measurements and bounding boxes
            were obtained.
        cost_matrix (ndarray): The cost matrix computed for the Hungarian algorithm.

    Returns:
        None
    """
    if cost_matrix is not None:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for r, c in zip(row_ind, col_ind):
            kf_objects[r].update(measurements[c])
            kf_objects[r].set_color_histogram(frame, bounding_boxes[c])

    while len(kf_objects) < len(measurements):
        kf = KalmanFilter()
        kf.set_color_histogram(frame, bounding_boxes[len(kf_objects)])
        kf_objects.append(kf)


def draw_predictions(frame: np.ndarray, kf_objects: list, ball_positions: dict) -> None:
    """
    Zeichnet vorhergesagte Positionen von Bällen auf dem Frame mit zusätzlichen visuellen Effekten.
    """
    colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0)]

    for i, kf in enumerate(kf_objects):
        predicted = kf.predict()
        px, py = int(predicted[0]), int(predicted[1])

        color = colors[i % len(colors)]
        if px == 0 or py == 0:
            continue
        cv2.circle(frame, (px, py), 10, color, -1)
        cv2.putText(frame, f"Ball {i + 1}", (px + 10, py + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # show speed of the ball
        speed_text = f"{kf.velocity:.2f} px/frame"
        cv2.putText(frame, speed_text, (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if i not in ball_positions:
            ball_positions[i] = []
        ball_positions[i].append((px, py))

        overlay = frame.copy()
        for j in range(1, len(ball_positions[i])):
            cv2.line(overlay, ball_positions[i][j - 1], ball_positions[i][j], color, 2)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
