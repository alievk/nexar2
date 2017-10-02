import numpy as np

import argparse
import Queue
import threading
import time
import os
import sys
sys.path.insert(0, './caffe-ssd-nexar/python')
import caffe
import pandas as pd
from collections import defaultdict
import imgaug as ia
from imgaug import augmenters as iaa
import cv2
from bbox_util import *

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--model')
parser.add_argument('--def', dest='model_def')
parser.add_argument('--val')
parser.add_argument('--from', dest='from_ind', default=0, type=int)
parser.add_argument('--to', dest='to_ind', default=-1, type=int)
parser.add_argument('--conf', default=0.08, type=float)
parser.add_argument('--sfx', default='')
parser.add_argument('--no-header', action='store_true')

args = parser.parse_args()

caffe.set_device(args.gpu)
caffe.set_mode_gpu()

val_image_root = os.path.abspath('./data/train')
test_image_root = os.path.abspath('./data/test')
image_root = val_image_root if args.val else test_image_root
get_image_path = lambda image_name: os.path.join(image_root, image_name)

conf_thresh = args.conf
box_size_thresh = 12
W, H = 1280, 720

save_img = False
bbox_draw_thresh = 0.1

use_caffe_preproc = False
print_time = False

model_def = args.model_def
model_weights = args.model

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

image_shape = net.blobs['data'].data.shape[2:]

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
BGR_MEAN = [104,117,123]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel

def preproc_batch(img_list):
    img_list_prep = []
    for img in img_list:
        img = cv2.resize(img, (image_shape)) - BGR_MEAN
        img = np.transpose(img, axes=[2, 0, 1])
        img = np.expand_dims(img, axis=0)
        img_list_prep.append(img)
    return np.concatenate(img_list_prep, axis=0)

class ThreadWorker(threading.Thread):
    stopAllThreads = False
    inputLock = threading.Condition()
    new_output = threading.Condition()
    inputQueue = Queue.Queue()
    outputQueue = Queue.Queue()

    def __init__(self, id):
        threading.Thread.__init__(self)
        self.id = id

    def run(self):
        il = self.inputLock
        noc = self.new_output
        iq = self.inputQueue
        oq = self.outputQueue
        while not self.stopAllThreads:
            image_name = None
            il.acquire()
            if not iq.empty():
                image_name = iq.get()
            il.release()

            if image_name:
                # DO IMAGE PROCESSING HERE
                image_path = get_image_path(image_name)
                image = transformer.preprocess('data', caffe.io.load_image(image_path))

                noc.acquire()
                oq.put(image)
                noc.notify()
                noc.release()

def augment(image_name):
    # DO IMAGE PROCESSING HERE
    image_path = get_image_path(image_name)
    image = cv2.imread(image_path)

    # AUGMENTATIONS

    image_aug = [{'image': image, 'scale': 0., 'flip': False}]

    zoom_in = -0.2
    cropnpad = iaa.CropAndPad(
        percent=zoom_in,
        pad_mode='constant',
        pad_cval=128,
        deterministic=True
    )
    image_zoom_in = cropnpad.augment_image(image)
    image_aug.append({'image': image_zoom_in, 'scale': zoom_in, 'flip': False})

    zoom_out = 0.2
    cropnpad = iaa.CropAndPad(
        percent=zoom_out,
        pad_mode='constant',
        pad_cval=128,
        deterministic=True
    )
    image_zoom_out = cropnpad.augment_image(image)
    image_aug.append({'image': image_zoom_out, 'scale': zoom_out, 'flip': False})

    fliplr = iaa.Fliplr(p=1, deterministic=True)
    # image_fliplr = fliplr.augment_image(image_zoom_in)
    # image_aug.append({'image': image_fliplr, 'scale': zoom_in, 'flip': True})
    image_fliplr = fliplr.augment_image(image_zoom_out)
    image_aug.append({'image': image_fliplr, 'scale': zoom_out, 'flip': True})

    return image_aug

def coord_func(xmin, xmax, dim, aug_scale, flip):
    scale = (1. + 2. * aug_scale)
    xmin_ = (xmin * scale - aug_scale) * dim
    xmax_ = (xmax * scale - aug_scale) * dim
    if flip:
        xmin__ = xmin_
        xmin_ = dim - xmax_
        xmax_ = dim - xmin__
    return int(round(xmin_)), int(round(xmax_))

# #######################################################################################
# EVALUATE
# #######################################################################################
if args.val:
    test_anno_df = pd.read_csv(args.val)
    test_image_names = list(set(test_anno_df['image_filename']))
else:
    test_image_names = open(os.path.join(test_image_root, 'names.txt')).readlines()
    test_image_names = [name.strip() for name in test_image_names]

# threadWorkers = []
# for tid in range(batch_size):
#     tw = ThreadWorker(tid)
#     tw.start()
#     threadWorkers.append(tw)

det_dict = defaultdict(list)

assert args.to_ind == -1 or args.from_ind < args.to_ind
img_ind_from = args.from_ind
img_ind_to = args.to_ind if args.to_ind > 0 else len(test_image_names)
img_ind_to = min(img_ind_to, len(test_image_names))

print 'Detecting from {} to {}...'.format(img_ind_from, img_ind_to)
start_total = time.time()
for img_ind in xrange(img_ind_from, img_ind_to):
    start_load = time.time()
    # image_aug_batch = []
    # remainder = max_val_num - img_ind
    # actual_batch = remainder if remainder < batch_size else batch_size
    # il = ThreadWorker.inputLock
    # iq = ThreadWorker.inputQueue
    # for bi in range(actual_batch):
    #     image_name = val_names[img_ind]
    #
    #     il.acquire()
    #     iq.put(image_name)
    #     il.release()
    #
    #     img_ind += 1
    #
    # noc = ThreadWorker.new_output
    # oq = ThreadWorker.outputQueue
    # noc.acquire()
    # while oq.qsize() != actual_batch:
    #     noc.wait()
    # while not oq.empty():
    #     image_aug = oq.get()
    #     image_aug_batch.append(image_aug)
    # assert oq.empty()
    # noc.release()

    image_name = test_image_names[img_ind]
    image_aug = augment(image_name)

    t_load = time.time() - start_load
    if print_time: print 'load time: %f' % t_load

    # assert len(image_aug_batch) == actual_batch

    # img_batch = np.concatenate(batch_img_lst, axis=0)
    # if actual_batch != batch_size: # should happen once
    #     print 'not complete batch: ', actual_batch
    #     net.blobs['data'].reshape(actual_batch, 3, image_shape[0], image_shape[1])

    start_prep = time.time()
    if use_caffe_preproc:
        image_batch = np.concatenate(
            [np.expand_dims(transformer.preprocess('data', aug['image']), axis=0) for aug in image_aug],
            axis=0
        )
    else:
        image_batch = preproc_batch([aug['image'] for aug in image_aug])
    t_prep = time.time() - start_prep
    if print_time: print "preproc time: %f" % t_prep

    if image_batch.shape[0] != net.blobs['data'].data.shape[0]:
        net.blobs['data'].reshape(image_batch.shape[0], 3, image_shape[0], image_shape[1])

    start_detect = time.time()
    net.blobs['data'].data[...] = image_batch
    detections = net.forward()['detection_out']
    t_detect = time.time() - start_detect
    if print_time: print 'detect time: %f' % t_detect

    start_filter = time.time()

    conf_mask = detections[0,0,:,2] >= conf_thresh
    det_img  = detections[0,0,conf_mask,0].astype(int)
    det_conf = detections[0,0,conf_mask,2]
    det_xmin = detections[0,0,conf_mask,3]
    det_ymin = detections[0,0,conf_mask,4]
    det_xmax = detections[0,0,conf_mask,5]
    det_ymax = detections[0,0,conf_mask,6]

    if save_img:
        img_boxes_before_nms = np.copy(image_aug[0]['image'])
        img_boxes_after_nms = np.copy(image_aug[0]['image'])

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

    boxes_list = []
    conf_list = []
    for box_ind in xrange(det_img.size):
        img_batch_ind = det_img[box_ind]
        scale = image_aug[img_batch_ind]['scale']
        flip = image_aug[img_batch_ind]['flip']
        x0, x1 = coord_func(det_xmin[box_ind], det_xmax[box_ind], W, scale, flip)
        y0, y1 = coord_func(det_ymin[box_ind], det_ymax[box_ind], H, scale, False)
        conf = det_conf[box_ind]

        if save_img and conf > bbox_draw_thresh:
            img_boxes_before_nms = draw_boxes(img_boxes_before_nms, [[x0, y0, x1, y1]], [conf], colors[img_batch_ind])

        # skip those prediction boxes which where taken from zoomed in images and are close to borders
        if scale < 0.:
            th = 10
            if x0 < th or x1 > W-th or \
                y0 < th or y1 > H-th:
                continue

        bw, bh = x1 - x0, y1 - y0
        if not (#0 <= x0 < W and 0 <= x1 < W and 0 <= y0 < H and 0 <= y1 <= H and
                bw >= box_size_thresh and bh >= box_size_thresh):
            continue

        boxes_list.append([x0, y0, x1, y1])
        conf_list.append(conf)

    if not boxes_list:
        print image_name + ' - no det'
        continue

    nms_iou_thresh = 0.75
    box_avg_type = 'mean'
    nms_boxes, nms_conf = nms_custom(np.array(boxes_list), np.array(conf_list), nms_iou_thresh, box_avg_type)

    num_final_boxes = nms_boxes.shape[0]
    for i in xrange(num_final_boxes):
        x0, y0, x1, y1 = nms_boxes[i]
        conf = nms_conf[i]
        det_dict['x0'].append(x0)
        det_dict['y0'].append(y0)
        det_dict['x1'].append(x1)
        det_dict['y1'].append(y1)
        det_dict['confidence'].append(conf)
        det_dict['image_filename'].append(image_name)
        det_dict['label'].append('car')

        if save_img and conf > bbox_draw_thresh:
            img_boxes_after_nms = draw_boxes(img_boxes_after_nms, [[x0, y0, x1, y1]], [conf], (255, 255, 255))

    t_filter = time.time() - start_filter
    if print_time: print 'filter time: %f' % t_filter

    if save_img:
        cv2.imwrite(os.path.join('out', image_name), img_boxes_before_nms)
        cv2.imwrite(os.path.join('out', image_name+'_.jpg'), img_boxes_after_nms)

    if img_ind > 0 and img_ind % 100 == 0:
        print('%s/%s'%(img_ind, img_ind_to-1))

t_total = time.time() - start_total

if args.val:
    val_name = os.path.splitext(args.val)[0]
    dt_csv = 'val_{}_{}-{}_{}.csv'.format(val_name, img_ind_from, img_ind_to, args.sfx)
else:
    dt_csv = 'test_{}-{}_{}.csv'.format(img_ind_from, img_ind_to, args.sfx)
dt_csv = os.path.join('out', dt_csv)

det_df = pd.DataFrame(det_dict, columns=['image_filename', 'x0', 'y0', 'x1', 'y1', 'label', 'confidence'])
det_df.to_csv(dt_csv, index=False, header=not args.no_header)

print('Detection written to: ' + dt_csv)
print('Time: %f' % t_total)

# ThreadWorker.stopAllThreads = True
# for tw in threadWorkers:
#     tw.join()

# evaluate detections
if args.val:
    sys.path.append('challenge2-evaluation/evaluate')
    import eval_challenge
    eval_challenge.DEBUG = True
    iou_threshold = 0.75
    print ('AP: {}'.format(eval_challenge.eval_detector_csv(args.val, dt_csv, iou_threshold)))