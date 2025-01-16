import os
import numpy as np
import cv2
from ultralytics import YOLO


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
        conf_threshold: The confidence threshold for the object detection.

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
