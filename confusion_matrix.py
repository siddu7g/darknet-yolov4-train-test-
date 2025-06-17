import os
import cv2
from collections import defaultdict

# Define paths
ground_truth_dir = '/home/data-center/Traffic_Lights_yolov12/classify_traffic_lights/val'  # your makesense .txt files
prediction_dir = '/home/data-center/Traffic_Pred'     # your parsed prediction .txt files
image_dir = '/home/data-center/Traffic_Lights_yolov12/classify_traffic_lights/val'         # folder of original images

IOU_THRESHOLD = 0.25
class_map = {'Red_Light': 0, 'Yellow_Light': 1, 'Green_Light': 2}
id_to_class = {v: k for k, v in class_map.items()}

def iou(box1, box2):
    def to_coords(box):
        x, y, w, h = box
        return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
    b1 = to_coords(box1)
    b2 = to_coords(box2)

    xi1 = max(b1[0], b2[0])
    yi1 = max(b1[1], b2[1])
    xi2 = min(b1[2], b2[2])
    yi2 = min(b1[3], b2[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    b1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
    b2_area = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union_area = b1_area + b2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

tp_counter = defaultdict(int)
fp_counter = defaultdict(int)
fn_counter = defaultdict(int)

for file in os.listdir(ground_truth_dir):
    if not file.endswith('.txt'):
        continue

    image_name = file.replace('.txt', '')
    img_path = os.path.join(image_dir, image_name + '.jpg')
    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    # Load GT
    gt_path = os.path.join(ground_truth_dir, file)
    gt_boxes = []
    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            cls_id = int(parts[0])
            x, y, bw, bh = map(float, parts[1:])
            gt_boxes.append((cls_id, [x, y, bw, bh]))

    # Load predictions (with confidence)
    pred_path = os.path.join(prediction_dir, file)
    pred_boxes = []
    if os.path.exists(pred_path):
        with open(pred_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                cls_id = int(parts[0])
                x, y, bw, bh = map(float, parts[1:5])
                conf = float(parts[4])
                pred_boxes.append((cls_id, [x, y, bw, bh], conf))

    # Sort predictions by confidence descending
    pred_boxes.sort(key=lambda x: x[2], reverse=True)

    matched_gt = set()

    for pred_cls, pred_box, _conf in pred_boxes:
        best_iou = 0
        best_gt_idx = -1
        for idx, (gt_cls, gt_box) in enumerate(gt_boxes):
            if idx in matched_gt or pred_cls != gt_cls:
                continue
            iou_val = iou(pred_box, gt_box)
            if iou_val > best_iou:
                best_iou = iou_val
                best_gt_idx = idx

        if best_iou >= IOU_THRESHOLD:
            tp_counter[pred_cls] += 1
            matched_gt.add(best_gt_idx)
        else:
            fp_counter[pred_cls] += 1

    for idx, (gt_cls, _) in enumerate(gt_boxes):
        if idx not in matched_gt:
            fn_counter[gt_cls] += 1

print("Evaluation Results:")
all_classes = set(tp_counter.keys()) | set(fn_counter.keys())
for cls_id in sorted(all_classes):
    tp = tp_counter[cls_id]
    fp = fp_counter[cls_id]
    fn = fn_counter[cls_id]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"Class: {id_to_class[cls_id]}")
    print(f" TP: {tp}, FP: {fp}, FN: {fn}")
    print(f" Precision: {precision:.2f}, Recall: {recall:.2f}\n")
