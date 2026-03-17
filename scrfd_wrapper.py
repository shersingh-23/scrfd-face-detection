import cv2
import numpy as np
import onnxruntime as ort
from functools import lru_cache


class SCRFDDetectorMax:
    def __init__(self, model_path, det_size=(640, 640),
                 conf_thresh=0.5, iou_thresh=0.4):

        self.det_w, self.det_h = det_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider", "CUDAExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.strides = [8, 16, 32]
        self.num_anchors = 2
        self.mean = 127.5
        self.std = 128.0

        print("SCRFD Loaded Successfully")


    def preprocess(self, img):
        h0, w0 = img.shape[:2]

        resized = cv2.resize(img, (self.det_w, self.det_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        blob = (rgb.astype(np.float32) - self.mean) / self.std
        blob = blob.transpose(2, 0, 1)[None]

        return blob, w0, h0


    @lru_cache(maxsize=10)
    def make_anchors(self, fm_h, fm_w, stride):
        y, x = np.mgrid[:fm_h, :fm_w]
        anchors = np.stack([x, y], axis=-1).astype(np.float32)
        anchors = anchors.reshape(-1, 2) * stride
        anchors = np.repeat(anchors, self.num_anchors, axis=0)
        return anchors


    def decode_bbox(self, anchors, deltas):
        boxes = np.empty((len(deltas), 4), dtype=np.float32)
        boxes[:, 0] = anchors[:, 0] - deltas[:, 0]
        boxes[:, 1] = anchors[:, 1] - deltas[:, 1]
        boxes[:, 2] = anchors[:, 0] + deltas[:, 2]
        boxes[:, 3] = anchors[:, 1] + deltas[:, 3]
        return boxes

 
    def decode_kps(self, anchors, deltas):
        kps = np.empty((len(deltas), 5, 2), dtype=np.float32)
        kps[:, :, 0] = anchors[:, 0:1] + deltas[:, 0::2]
        kps[:, :, 1] = anchors[:, 1:2] + deltas[:, 1::2]
        return kps

    def nms(self, boxes, scores):
        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[1:][iou <= self.iou_thresh]

        return keep


    def adaptive_padding(self, box, img_w, img_h):
        x1, y1, x2, y2 = box

        face_w = x2 - x1
        face_h = y2 - y1
        pad = 0.15 * max(face_w, face_h)
        pad = min(pad, 0.08 * max(img_w, img_h))

        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(img_w - 1, x2 + pad)
        y2 = min(img_h - 1, y2 + pad)

        return np.array([x1, y1, x2, y2], dtype=np.float32)


    def detect(self, img):
        blob, w0, h0 = self.preprocess(img)

        outs = self.session.run(self.output_names, {self.input_name: blob})

        scores_list = outs[0:3]
        bbox_list = outs[3:6]
        kps_list = outs[6:9]

        all_boxes, all_scores, all_kps = [], [], []

        for i, stride in enumerate(self.strides):
            scores = scores_list[i].reshape(-1)

            bbox_preds = bbox_list[i].reshape(-1, 4) * stride
            kps_preds = kps_list[i].reshape(-1, 10) * stride

            fm_h = self.det_h // stride
            fm_w = self.det_w // stride
            anchors = self.make_anchors(fm_h, fm_w, stride)

            mask = scores >= self.conf_thresh
            if not mask.any():
                continue

            scores = scores[mask]
            bbox_preds = bbox_preds[mask]
            kps_preds = kps_preds[mask]
            anchors = anchors[mask]

            boxes = self.decode_bbox(anchors, bbox_preds)
            kps = self.decode_kps(anchors, kps_preds)

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_kps.append(kps)

        if not all_boxes:
            return []

        all_boxes = np.vstack(all_boxes)
        all_scores = np.hstack(all_scores)
        all_kps = np.vstack(all_kps)

        sx, sy = w0 / self.det_w, h0 / self.det_h
        all_boxes[:, [0, 2]] *= sx
        all_boxes[:, [1, 3]] *= sy
        all_kps[:, :, 0] *= sx
        all_kps[:, :, 1] *= sy

        keep = self.nms(all_boxes, all_scores)

        detections = []
        for i in keep:
            padded_box = self.adaptive_padding(all_boxes[i], w0, h0)

            detections.append({
                "conf": float(all_scores[i]),
                "bbox": padded_box, 
                "keypoints": all_kps[i]
            })

        return detections