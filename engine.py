# This file is adapted from the official PyTorch GitHub repository:
# https://github.com/pytorch/vision/tree/main/references/detection
# Description:
# Implements the core training and evaluation loops for object detection models. 
# It includes functions for training each epoch, calculating losses, and saving the model.
#
# This file is essential for managing our training workflow and ensuring model consistency.


import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate_map(model, data_loader, device, iou_threshold=0.5):
    model.eval()
    cpu_device = torch.device("cpu")

    all_detections = []
    all_ground_truths = []

    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        for target, output in zip(targets, outputs):
            image_id = target["image_id"].item()
            gt_boxes = target["boxes"].tolist()
            pred_boxes = output["boxes"].tolist()
            pred_scores = output["scores"].tolist()
            all_ground_truths.extend([(image_id, box) for box in gt_boxes])
            all_detections.extend([(image_id, box, score) for box, score in zip(pred_boxes, pred_scores)])

    ap = calculate_map(all_detections, all_ground_truths, iou_threshold)
    print(f"mAP at IoU threshold {iou_threshold}: {ap:.4f}")
    return ap


def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)

    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def calculate_map(detections, ground_truths, iou_threshold):
    detections.sort(key=lambda x: x[2], reverse=True)
    image_to_gt = {}
    for image_id, gt_box in ground_truths:
        image_to_gt.setdefault(image_id, []).append(gt_box)

    tp = 0
    fp = 0
    total_gt_boxes = len(ground_truths)
    precisions = []
    recalls = []

    for image_id, pred_box, _ in detections:
        max_iou = 0
        best_gt_idx = -1
        gt_boxes = image_to_gt.get(image_id, [])

        for idx, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(pred_box, gt_box)
            if iou > max_iou:
                max_iou = iou
                best_gt_idx = idx

        if max_iou >= iou_threshold:
            tp += 1
            del gt_boxes[best_gt_idx]
        else:
            fp += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / total_gt_boxes if total_gt_boxes > 0 else 0
        precisions.append(precision)
        recalls.append(recall)

    ap = 0.0
    for i in range(1, len(recalls)):
        ap += precisions[i] * (recalls[i] - recalls[i - 1])

    return ap
