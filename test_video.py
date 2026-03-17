import cv2
import numpy as np
from scrfd_wrapper2 import SCRFDDetectorMax
 
 
def bbox_color(conf):
    conf_percent = conf * 100
 
    if conf_percent >= 90:
        return (0, 255, 0)        
    elif conf_percent >= 70:
        return (0, 165, 255)    
    else:
        return (0, 0, 255)       
 
 
if __name__ == "__main__":
 
    MODEL_PATH = "./models/det_10g.onnx"
    VIDEO_PATH = "test_192 1.mp4"
    OUTPUT_PATH = "./output/output_video_max.mp4"
 
    detector = SCRFDDetectorMax(
        MODEL_PATH,
        det_size=(640, 640),
        conf_thresh=0.5,
        iou_thresh=0.2
    )
 
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Could not open video")
        exit()
 
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
 
    frame_id = 0
 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
 
        detections = detector.detect(frame)
 
        for det in detections:
            conf = det["conf"]
            x1, y1, x2, y2 = map(int, det["bbox"])
            kps = det["keypoints"]
 
            color = bbox_color(conf)
 
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
 
            label = f"Face {conf*100:.1f}%"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
 
            for (kx, ky) in kps:
                cv2.circle(frame, (int(kx), int(ky)), 2, (255, 0, 0), -1)
 
        out.write(frame)
        frame_id += 1
        print(f"\rTotal frames processed: {frame_id}", end="", flush=True)
 
    cap.release()
    out.release()
 
    print("\nVideo saved to:", OUTPUT_PATH)
