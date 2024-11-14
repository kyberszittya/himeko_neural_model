import ultralytics as ul
import cv2

def main():
    # Load the pose detection YOLOv11 model
    pose_detection = ul.YOLO("yolo11n-pose.pt")
    # Load the video stream (camera)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            results = pose_detection(frame)
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLO Inference", annotated_frame)
            # Display the results
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == "__main__":
    main()
