import numpy as np
import cv2
import os

def batch_iou(boxes, box):
    lr = np.maximum(
        np.minimum(boxes[:, 2], box[2]) - np.maximum(boxes[:, 0], box[0]),
        0.
    )
    tb = np.maximum(
        np.minimum(boxes[:, 3], box[3]) - np.maximum(boxes[:, 1], box[1]),
        0.
    )
    inter = lr * tb
    union = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) +\
            (box[2] - box[0]) * (box[3] - box[1]) - inter
    return inter.astype(np.float32) / union


def nms(boxes, probs, iou_thresh):
    order = probs.argsort()[::-1]
    keep = [True] * len(order)

    for i in range(len(order) - 1):
        ovps = batch_iou(boxes[order[i + 1:]], boxes[order[i]])
        for j, ov in enumerate(ovps):
            if ov > iou_thresh:
                keep[order[j + i + 1]] = False
    return keep

def nms_custom(boxes, conf, iou_thresh, box_avg_type='mean'):
    order = conf.argsort()[::-1]
    boxes_ord = boxes[order]
    conf_ord = conf[order]

    num_boxes = boxes.shape[0]
    fin_boxes = []
    fin_conf = []
    boxes_left = boxes_ord
    conf_left = conf_ord
    while boxes_left.shape[0] > 1:
        ovps = batch_iou(boxes_left[1:], boxes_left[0])
        mask = ovps > iou_thresh

        boxes_group = [boxes_left[0].reshape(1,4)]
        conf_group = [conf_left[0]]
        has_group = mask.sum() > 0
        if has_group:
            boxes_group.append(boxes_left[1:][mask].reshape(-1,4))
            conf_group.extend(conf_left[1:][mask])
        boxes_group = np.concatenate(boxes_group, axis=0)
        conf_group = np.array(conf_group)

        if box_avg_type == 'conf':
            box_weights = conf_group / conf_group.sum()
            box_avg = (boxes_group * box_weights[:,np.newaxis]).sum(axis=0)
        elif box_avg_type == 'conf_pow':
            box_weights = conf_group ** 4
            box_weights /= box_weights.sum()
            box_avg = (boxes_group * box_weights[:,np.newaxis]).sum(axis=0)
        elif box_avg_type == 'mean':
            box_avg = boxes_group.mean(axis=0)
        else:
            raise ValueError
        conf_avg = conf_group.max()
        fin_boxes.append(box_avg.reshape(1,4))
        fin_conf.append(conf_avg)

        keep = np.bitwise_not(mask)
        conf_left = conf_left[1:][keep]
        boxes_left = boxes_left[1:][keep]

    if boxes_left.shape[0] == 1:
        fin_boxes.append(boxes_left[0].reshape(1,4))
        fin_conf.append(conf_left[0])

    fin_boxes = np.round(np.concatenate(fin_boxes, axis=0)).astype(np.int32)
    fin_conf = np.array(fin_conf)
    return fin_boxes, fin_conf

def filter_predictions(prob, cls, boxes, num_classes, num_top_detections, class_prob_thresh, nms_iou_thresh):
    assert prob.ndim == 1
    assert cls.ndim == 1
    assert boxes.ndim == 2

    if len(prob) > num_top_detections and num_top_detections > 0:
        prob_mask = np.argsort(prob)[:-num_top_detections - 1:-1]
    else:
        prob_mask = np.nonzero(prob > class_prob_thresh)[0]
    prob = prob[prob_mask]
    cls = cls[prob_mask]
    boxes = boxes[prob_mask]

    #     print prob
    #     print cls
    #     print boxes

    final_prob, final_cls, final_boxes = [], [], []

    for c in range(num_classes):
        cls_mask = np.nonzero(cls == c)[0]
        keep = nms(boxes[cls_mask], prob[cls_mask], nms_iou_thresh)
        for i, k in enumerate(keep):
            if k:
                final_boxes.append(boxes[cls_mask[i]])
                final_prob.append(prob[cls_mask[i]])
                final_cls.append(c)

    return final_prob, final_cls, final_boxes

def draw_boxes(im, box_list, conf_list, color=(0,255,0), cdict=None):
    im = im.copy()
    for bbox, conf in zip(box_list, conf_list):
        #cx, cy, w, h = bbox
        #xmin, ymin = int(cx-w/2), int(cy-h/2)
        #xmax, ymax = int(cx+w/2), int(cy+h/2)
        xmin, ymin, xmax, ymax = bbox

        # draw box
        cv2.rectangle(im, (xmin, ymin), (xmax, ymax), color, 1)
        # draw label
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im, str(conf), (xmin, ymax), font, 0.4, (255,255,255), 1)

    return im



