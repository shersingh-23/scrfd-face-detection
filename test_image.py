import cv2
import numpy as np
from scrfd_wrapper import SCRFDDetector


# ------------------------------------
# Color Rule for Confidence
# ------------------------------------
def bbox_color(conf):
    conf_percent = conf * 100

    if conf_percent >= 90:
        return (0, 255, 0)        # Green
    elif conf_percent >= 70:
        return (0, 165, 255)      # Orange
    else:
        return (0, 0, 255)        # Red


# ------------------------------------
# Non-Max Suppression (NMS)
# ------------------------------------
def nms(detections, iou_thresh=0.4):
    """
    detections: list of dicts {conf, bbox, keypoints}
    returns: filtered detections after NMS
    """

    if len(detections) == 0:
        return []

    boxes = np.array([det["bbox"] for det in detections])
    scores = np.array([det["conf"] for det in detections])

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return [detections[i] for i in keep]


# ------------------------------------
# MAIN TEST
# ------------------------------------
if __name__ == "__main__":

    MODEL_PATH = "./det_10g.onnx"
    IMAGE_PATH = "./TestImage.jpg"

    # Load detector
    detector = SCRFDDetector(MODEL_PATH, det_size=(640, 640))

    # Read image
    img = cv2.imread(IMAGE_PATH)

    if img is None:
        print("Error: Image not found!")
        exit()

    # Run detection
    detections = detector.detect(img, conf_thresh=0.5)

    print("Raw Faces Detected:", len(detections))

    # Apply NMS
    detections = nms(detections, iou_thresh=0.4)

    print("Final Faces After NMS:", len(detections))

    # Draw results
    for det in detections:

        conf = det["conf"]
        x1, y1, x2, y2 = map(int, det["bbox"])
        kps = det["keypoints"]

        color = bbox_color(conf)

        # Draw Bounding Box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Confidence Label
        label = f"Face {conf*100:.1f}%"
        cv2.putText(
            img,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        # Draw Keypoints
        for (kx, ky) in kps:
            cv2.circle(img, (int(kx), int(ky)), 3, (255, 0, 0), -1)

    # ✅ Save output (No imshow needed)
    output_path = "output_result.jpg"
    cv2.imwrite(output_path, img)

    print("Saved Output Image:", output_path)