import cv2
import numpy as np
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from data.data import FILE_PATH
from kalman_filter import KalmanFilter


# Tracking-Methode
def ball_analyser(video_path):
    model = YOLO("yolov8n.pt", verbose=False)
    cap = cv2.VideoCapture(video_path)

    # Set up output video writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output_video.mp4", fourcc, fps, (frame_width, frame_height))

    kf_objects = []
    ball_positions = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        measurements = []
        bounding_boxes = []

        for box in results[0].boxes.cpu().numpy():
            cls = int(box.cls[0])
            cls_name = model.names[cls]

            if cls_name != "sports ball" or box.conf[0] < 0.75:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            measurements.append(np.array([[np.float32(cx)], [np.float32(cy)]]))
            bounding_boxes.append((x1, y1, x2, y2))

        # Kostenmatrix erstellen
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

        if cost_matrix:
            cost_matrix = np.array(cost_matrix)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for r, c in zip(row_ind, col_ind):
                kf_objects[r].update(measurements[c])
                kf_objects[r].set_color_histogram(frame, bounding_boxes[c])

        while len(kf_objects) < len(measurements):
            kf = KalmanFilter()
            kf.set_color_histogram(frame, bounding_boxes[len(kf_objects)])
            kf_objects.append(kf)

        for i, kf in enumerate(kf_objects):
            predicted = kf.predict()

            if i not in ball_positions:
                ball_positions[i] = []
            ball_positions[i].append((int(predicted[0]), int(predicted[1])))

            px, py = int(predicted[0]), int(predicted[1])
            cv2.circle(frame, (px, py), 10, (0, 255, 0), -1)
            cv2.putText(frame, f"Ball {i + 1}", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        for i, positions in ball_positions.items():
            for j in range(1, len(positions)):
                if positions[j - 1] == (0, 0) or positions[j] == (0, 0):
                    continue
                cv2.circle(frame, positions[j], 2, (0, 0, 0), -1)

        out.write(frame)  # Write the frame to the output video
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    out.release()  # Release the video writer
    cv2.destroyAllWindows()


# Video Pfad
ball_analyser(FILE_PATH)

