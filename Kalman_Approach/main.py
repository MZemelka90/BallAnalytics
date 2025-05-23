import cv2
from ultralytics import YOLO
from kalman_filter import create_cost_matrix, update_kalman_filters, draw_predictions
from helper_functions.helper import (
    get_file_path_in_project,
    draw_trail,
    initialize_video_writer,
    extract_ball_positions_and_bounding_boxes,
    draw_ball_statistics
)


def ball_analyser(video_path: str) -> None:
    model = YOLO("yolov8n.pt", verbose=False)
    cap = cv2.VideoCapture(video_path)
    out = initialize_video_writer(cap, output_path=get_file_path_in_project("examples", "result.mp4"))

    kf_objects = []
    ball_positions = {}
    frame_counter = 0
    ball_detections = {}

    while cap.isOpened():
        frame_counter += 1
        ret, frame = cap.read()
        if not ret:
            break

        measurements, bounding_boxes = extract_ball_positions_and_bounding_boxes(model, frame, ball_detections, 0.7)
        cost_matrix = create_cost_matrix(kf_objects, measurements, bounding_boxes, frame)
        update_kalman_filters(kf_objects, measurements, bounding_boxes, frame, cost_matrix)

        draw_predictions(frame, kf_objects, ball_positions)
        draw_ball_statistics(frame, ball_detections, frame_counter)
        draw_trail(frame, ball_positions)

        out.write(frame)
        cv2.imshow("Football Ball Analyser", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ball_analyser(get_file_path_in_project("examples", "Example_Video.mp4"))
