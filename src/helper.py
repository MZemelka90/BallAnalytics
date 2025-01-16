import os
import numpy as np
import cv2
from ultralytics import YOLO
from kalman_filter import KalmanFilter
from scipy.optimize import linear_sum_assignment



def get_file_path_in_project(dir_name: str, file_name: str) -> str:
    """
    Get the absolute path of a file in the project directory.

    Args:
        dir_name (str): The name of the directory in the project root where the file is located.
        file_name (str): The name of the file.

    Returns:
        str: The absolute path of the file.
    """
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_dir, dir_name, file_name)


def draw_trail(frame: np.ndarray, ball_positions: dict) -> None:
    """
    Draw trails of balls based on previous positions.

    Args:
        frame (ndarray): The frame to draw on.
        ball_positions (dict): A dictionary of ball positions.
        The keys are ball IDs and the values are lists of positions.
    """
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    for i, positions in ball_positions.items():
        color = colors[i % len(colors)]
        for j in range(1, len(positions)):
            if positions[j - 1] == (0, 0) or positions[j] == (0, 0):
                continue
            cv2.circle(frame, positions[j], 2, color, -1)


def draw_predictions(frame: np.ndarray, kf_objects: list, ball_positions: dict) -> None:
    """
    Draw predicted positions of balls on the given frame.

    Args:
        frame (np.ndarray): The frame to draw on.
        kf_objects (list): A list of KalmanFilter objects used for prediction.
        ball_positions (dict): A dictionary to store ball positions. The keys are ball IDs and
        the values are lists of positions.

    Returns:
        None
    """
    for i, kf in enumerate(kf_objects):
        predicted = kf.predict()

        if i not in ball_positions:
            ball_positions[i] = []
        ball_positions[i].append((int(predicted[0]), int(predicted[1])))

        px, py = int(predicted[0]), int(predicted[1])
        cv2.circle(frame, (px, py), 10, (0, 255, 0), -1)
        cv2.putText(frame, f"Ball {i + 1}", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


def initialize_video_writer(cap: cv2.VideoCapture, output_path: str) -> cv2.VideoWriter:
    """
    Initializes a VideoWriter from a given VideoCapture object and an output path.

    Args:
        cap: The VideoCapture object to get the video properties from.
        output_path: The path to save the video to.

    Returns:
        A VideoWriter object for the given output path.
    """
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    return cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


def extract_ball_positions_and_bounding_boxes(model: YOLO, frame: np.ndarray, conf_threshold: float = 0.75) -> tuple:
    """
    Process a frame from a video and find all detected balls in the frame.

    Args:
        model: The YOLO model used for object detection.
        frame: The frame to process.

    Returns:
        A tuple of two lists. The first list contains the center positions of the
        detected balls as numpy arrays. The second list contains the bounding boxes
        of the detected balls as tuples of four integers (x1, y1, x2, y2).
    """
    results = model(frame, verbose=False)
    measurements = []
    bounding_boxes = []

    if results is not None and len(results) > 0:
        for box in results[0].boxes.cpu().numpy():
            cls = int(box.cls[0])
            cls_name = model.names[cls]

            if cls_name != "sports ball" or box.conf[0] < conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            measurements.append(np.array([[np.float32(cx)], [np.float32(cy)]]))
            bounding_boxes.append((x1, y1, x2, y2))

    return measurements, bounding_boxes


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
