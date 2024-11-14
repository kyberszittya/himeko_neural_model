import ultralytics as ul
import cv2
from collections import defaultdict
import numpy as np


def main():
    track_history = defaultdict(lambda: [])
    # Load the pose detection YOLOv11 model
    pose_detection = ul.YOLO("yolo11n-pose.pt")
    # Load the video stream (camera)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            results = pose_detection.track(frame, persist=True)
            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Plot the tracks
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 90 tracks for 90 frames
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

                # Display the annotated frame
                cv2.imshow("YOLO11 Tracking", annotated_frame)
            else:
                # Display the annotated frame
                cv2.imshow("YOLO11 Tracking", results[0].plot())

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    main()
