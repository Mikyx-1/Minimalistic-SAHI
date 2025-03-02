import numpy as np
from ultralytics import YOLO


class SAHI:
    def __init__(self, weight_dir: str) -> None:
        self.model = YOLO(weight_dir)

    @staticmethod
    def slice_image(
        image: np.ndarray,
        slice_height: int,
        slice_width: int,
        overlap_height_ratio: float,
        overlap_width_ratio: float,
    ):

        img_height, img_width = image.shape[:2]
        step_h = int(slice_height * (1 - overlap_height_ratio))
        step_w = int(slice_width * (1 - overlap_width_ratio))

        slices = []

        for y in range(0, img_height - slice_height + 1, step_h):
            for x in range(0, img_width - slice_width + 1, step_w):
                patch = image[y : y + slice_height, x : x + slice_width]
                slices.append((patch, y, x))

        # Handle right and bottom edge cases
        if img_height % step_h != 0:
            for x in range(0, img_width - slice_width + 1, step_w):
                patch = image[
                    img_height - slice_height : img_height, x : x + slice_width
                ]
                slices.append((patch, img_height - slice_height, x))

        if img_width % step_w != 0:
            for y in range(0, img_height - slice_height + 1, step_h):
                patch = image[y : y + slice_height, img_width - slice_width : img_width]
                slices.append((patch, y, img_width - slice_width))

        # Bottom-right corner
        if img_height % step_h != 0 and img_width % step_w != 0:
            patch = image[
                img_height - slice_height : img_height,
                img_width - slice_width : img_width,
            ]
            slices.append((patch, img_height - slice_height, img_width - slice_width))

        return slices

    @staticmethod
    def custom_nms(boxes, scores, iou_threshold=0.5):
        indices = []
        sorted_indices = scores.argsort()[::-1]

        while len(sorted_indices) > 0:
            current = sorted_indices[0]
            indices.append(current)
            if len(sorted_indices) == 1:
                break

            current_box = boxes[current]
            other_boxes = boxes[sorted_indices[1:]]

            x1 = np.maximum(current_box[0], other_boxes[:, 0])
            y1 = np.maximum(current_box[1], other_boxes[:, 1])
            x2 = np.minimum(current_box[2], other_boxes[:, 2])
            y2 = np.minimum(current_box[3], other_boxes[:, 3])

            intersection = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)
            box_area = (current_box[2] - current_box[0] + 1) * (
                current_box[3] - current_box[1] + 1
            )
            other_areas = (other_boxes[:, 2] - other_boxes[:, 0] + 1) * (
                other_boxes[:, 3] - other_boxes[:, 1] + 1
            )

            union = box_area + other_areas - intersection
            iou = intersection / union

            sorted_indices = sorted_indices[1:][iou <= iou_threshold]

        return indices

    @staticmethod
    def standardize_and_nms(preds, iou_threshold=0.5):
        all_detections = []

        for pred, y, x in preds:
            if pred.numel() > 0:
                for detection in pred.cpu().numpy():
                    x1, y1, x2, y2, confidence, class_id = detection
                    all_detections.append(
                        [x1 + x, y1 + y, x2 + x, y2 + y, confidence, class_id]
                    )

        if not all_detections:
            return []

        all_detections = np.array(all_detections)

        boxes = all_detections[:, :4]
        scores = all_detections[:, 4]
        class_ids = all_detections[:, 5]

        keep_indices = SAHI.custom_nms(boxes, scores, iou_threshold)

        final_detections = all_detections[keep_indices]

        return final_detections

    def run(
        self,
        image: np.ndarray,
        slice_height: int,
        slice_width: int,
        overlap_height_ratio: float,
        overlap_width_ratio: float,
    ):

        sliced_images = SAHI.slice_image(
            image, slice_height, slice_width, overlap_height_ratio, overlap_width_ratio
        )

        preds = []
        for sliced_image in sliced_images:
            patch, y, x = sliced_image
            pred = self.model(patch, verbose=False)[0].boxes.data

            if pred.numel() > 0:
                preds.append(((pred, y, x)))

        final_preds = SAHI.standardize_and_nms(preds)
        return final_preds
