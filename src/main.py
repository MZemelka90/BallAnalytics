import cv2
import numpy as np
import os
from ultralytics import YOLO
from data.data import FILE_PATH

model = YOLO("yolov8n.pt")

current_dir = os.path.dirname(os.path.abspath(__file__))
video_path = FILE_PATH
cap = cv2.VideoCapture(video_path)

# configuration of the output
output_path = os.path.join(current_dir, "output.mp4")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 3)
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 3)
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

if not cap.isOpened():
    print("Error: Video could not be opened.")
    exit()

print(f"Video Information: Width={frame_width}, Height={frame_height}, FPS={fps}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # scale frame
    frame = cv2.resize(frame, (frame_width, frame_height))

    # YOLO object detection on current frame
    results = model(frame)

    if not results:
        continue

    for result in results:
        boxes = result.boxes.cpu().numpy()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = int(box.cls[0])
            cls_name = model.names[cls]  # class name, we are looking for "sports ball"

            # only filter for sports balls
            if cls_name != "sports ball":
                continue
            # draw bounding box
            color = (0, 255, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{cls_name}_{i} {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out.write(frame)

    cv2.imshow("Football Analysis", frame)
    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break

# Ressourcen freigeben
cap.release()
out.release()
cv2.destroyAllWindows()