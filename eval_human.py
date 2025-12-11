import os
import cv2
import numpy as np
from glob import glob

pred_dir = "./outputs/feature-2stage-human/test/epoch=57-step=6000"
gt_dir   = "./datasets_vggt/human/rgb_feature_langseg"

def calculate_iou(y_true, y_pred, num_classes=2):
    iou = []
    for i in range(num_classes):
        true_labels = (y_true == i)
        predicted_labels = (y_pred == i)
        intersection = np.logical_and(true_labels, predicted_labels).sum()
        union = np.logical_or(true_labels, predicted_labels).sum()
        if union == 0:
            iou.append(np.nan)
        else:
            iou.append(intersection / union)
    return np.nanmean(iou)

def to_binary_mask(mask):
    return (mask > 0).astype(np.uint8)

ious = []

all_preds = glob(os.path.join(pred_dir, "*.jpg.jpg"))

for pred_path in all_preds:
    base = os.path.basename(pred_path)
    frame_id = base.split(".")[0]   # frame_XXXXX

    gt_path = os.path.join(gt_dir, f"{frame_id}.png_feature_vis.png")
    if not os.path.exists(gt_path):
        print("GT not found:", gt_path)
        continue

    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    if pred is None or gt is None:
        print("Load fail:", pred_path)
        continue

    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

    pred_bin = to_binary_mask(pred)
    gt_bin = to_binary_mask(gt)

    iou = calculate_iou(gt_bin, pred_bin, num_classes=2)
    if not np.isnan(iou):
        ious.append(iou)

final_iou = np.mean(ious)


print("=======================================")
print("Class-agnostic IoU (binary using calculate_iou):", final_iou)
print("=======================================")