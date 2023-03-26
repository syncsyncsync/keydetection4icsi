#!/usr/bin/env python
# coding: utf-8
# Keypoint Detection on ICSI Dataset
# 2022-12-12: Added data augmentation
DEBUG_MODE = False

import rospy
import roslib
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

import os
import numpy as np
import cv2

import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import argparse
import time
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer, launch
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.video_visualizer import ColorMode
from detectron2.structures import Instances, BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper
from detectron2.data import build_detection_train_loader
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.data import transforms as T
from scipy.optimize import linear_sum_assignment

global pre_boxes
global pre_keys
global pre_scores
global count 
count = 0 
pre_boxes = []
pre_keys = []
pre_scores = []
setup_logger()
K50_1 = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x"
K50_3 = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x"
K101 = "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x"
KX101 = "COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x"

K50_3_path = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
K101_path = "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"
DETECT_config1 = "ICSI-DETECTION/SP_keypoint_rcnn_R_101_FPN_3x.yaml"
DETECT_config2 = "ICSI-DETECTION/SP_keypoint_rcnn_R_50_FPN_3x.yaml"

DEFAULT_FPS = 20 
TRAIN_WEIGHT_CONF = K50_3_path
PRETRAIN_WEIGHT_NAME  = TRAIN_WEIGHT_CONF[:-5]
SCALE= 4
# *********************************************
DATA_SET_NAME = "sp15"
# *********************************************

#  DATA K101_3
BASE_DIR = '/home/icsiauto/detectron2/20230104keypoints'
LABEL_DIR = "annotations/person_keypoints_default.json"
IMAGE_DIR = "images"

_SMALL_OBJECT_AREA_THRESH = 1000
_LARGE_MASK_AREA_THRESH = 120000
_OFF_WHITE = (1.0, 1.0, 240.0 / 255)

# BGR format
_BLACK = (0, 0, 0)
_RED = (0, 0, 1.0)
_BLUE = (1.0, 0, 0)
_KEYPOINT_THRESHOLD = 0.05

# ---------------------------------------
# Customize the training process
# ---------------------------------------
class MyTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        my_augmentations = [
        T.RandomBrightness(0.9, 1.1),
        T.RandomContrast(0.9, 1.1),
        T.RandomFlip(prob=0.5), 
        # T.ResizeScale((0.9, 4, 640, 640), interpolation=cv2.INTER_LINEAR),
        # T.RandomCrop_CategoryAreaConstraint("absolute", (256, 256), 0.8, 1.2),
        T.RandomCrop("absolute", (256, 256)),
        T.RandomRotation(angle=[-90, 90])
        ]
        mapper = DatasetMapper(cfg, is_train=True, augmentations=my_augmentations)
        return build_detection_train_loader(cfg, mapper=mapper)

# ---------------------------------------
# Customize the visualizer
# ---------------------------------------
class SPVisualizer(Visualizer):
    def __init__(self, img_rgb, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE):
        super().__init__(img_rgb, metadata, scale, instance_mode)
        self.keypoint_threshold = _KEYPOINT_THRESHOLD
    def draw_and_connect_keypoints(self, keypoints):
        """
        Draws keypoints of an instance and follows the rules for keypoint connections
        to draw lines between appropriate keypoints. This follows color heuristics for
        line color.

        Args:
            keypoints (Tensor): a tensor of shape (K, 3), where K is the number of keypoints
                and the last dimension corresponds to (x, y, probability).

        Returns:
            output (VisImage): image object with visualizations.
        """

        visible = {}
        keypoint_names = self.metadata.get("keypoint_names")
        keypoints = self._convert_keypoints(keypoints)
        for idx, keypoint in enumerate(keypoints):
            # draw keypoint
            x, y, prob = keypoint
            if prob > self.keypoint_threshold:
                if keypoint_names:
                    keypoint_name = keypoint_names[idx]
                    visible[keypoint_name] = (x, y)
                    if keypoint_name == "target":
                        #self.draw_text(keypoint_name, (x +5, y), color=_RED)
                        self.draw_circle((x, y), color=_RED)
                    elif keypoint_name == "head":
                        self.draw_circle((x, y), color=_BLACK, radius=20)
                    else:
                        #self.draw_text(keypoint_name, (x, y), color=_BLUE)
                        self.draw_circle((x, y), color=_BLUE)
                else:
                    self.draw_circle((x, y), color=_BLUE)

        if self.metadata.get("keypoint_connection_rules"):
            for kp0, kp1 in self.metadata.keypoint_connection_rules:
            #for kp0, kp1, kp2 in self.metadata.keypoint_connection_rules:
                if kp0 in visible and kp1 in visible:
                    x0, y0 = visible[kp0]
                    x1, y1 = visible[kp1]
                    color = _BLACK
                    #color = tuple(x / 255.0 for x in color)
                    self.draw_line([x0, x1], [y0, y1], color=color, linewidth=0.5)

        # draw lines from nose to mid-shoulder and mid-shoulder to mid-hip
        # Note that this strategy is specific to person keypoints.
        # For other keypoints, it should just do nothing
        return self.output


def register_dataset(name, annotaitions, images, force=False):
    try:
        DatasetCatalog.get(name)
        print("Dataset {} already registered".format(name))
        if force:
            DatasetCatalog.clear()
            register_coco_instances(name, {}, annotaitions, images)
        else:
            print("annotation and image paths are not checked")
    except:
        print("Registering dataset {}".format(name))
        register_coco_instances(name, {}, annotaitions, images)
    return name


def init_ros():
    global bridge
    global spc
    global twist
    global twist_n
    global phase_info
    global pub, pub_sp2, pub_sp3
    global pub_i
    global pub2, pub2_i

    rospy.init_node('sperm_detection2', anonymous=True)
    global pre_boxes
    phase_info  = 0
    pre_boxes = []

    twist = Twist()
    twist.linear.x = 0.0
    twist.linear.y = 0.0
    twist.linear.z = 0.0
    twist.angular.x = 0.0
    twist.angular.y = 0.0
    twist.angular.z = 0.0

    twist_n = Twist()
    twist_n.linear.x = 0.0
    twist_n.linear.y = 0.0
    twist_n.linear.z = 0.0
    twist_n.angular.x = 0.0
    twist_n.angular.y = 0.0
    twist_n.angular.z = 0.0

    pub = rospy.Publisher('sperm_detection_twist', Twist, queue_size=10)
    pub_i = rospy.Publisher('sperm_detection_image', Image, queue_size=10)
    pub_sp2 = rospy.Publisher('sperm_detection_twist', Twist, queue_size=10)
    pub_sp3 = rospy.Publisher('sperm_detection_twist', Twist, queue_size=10)

    pub2 = rospy.Publisher('needle_twist', Twist, queue_size=10)
    pub2_i = rospy.Publisher('needle_image', Image, queue_size=10)

    bridge = CvBridge()
    spc = 0


def iou(a, b):
    ax_mn, ay_mn, ax_mx, ay_mx = a[0], a[1], a[2], a[3]
    bx_mn, by_mn, bx_mx, by_mx = b[0], b[1], b[2], b[3]

    a_area = (ax_mx - ax_mn + 1) * (ay_mx - ay_mn + 1)
    b_area = (bx_mx - bx_mn + 1) * (by_mx - by_mn + 1)

    abx_mn = max(ax_mn, bx_mn)
    aby_mn = max(ay_mn, by_mn)
    abx_mx = min(ax_mx, bx_mx)
    aby_mx = min(ay_mx, by_mx)

    w = max(0, abx_mx - abx_mn + 1)
    h = max(0, aby_mx - aby_mn + 1)
    intersect = w * h

    iou = intersect / (a_area + b_area - intersect)
    return iou


def core_predict(im, MetadataCatalog, visualize=True):
    outputs = predictor(im)

    if visualize:
        v = SPVisualizer(im[:, :, ::-1],
                        metadata=MetadataCatalog.get(DATA_SET_NAME),
                        scale=1.5,
                        instance_mode=ColorMode.SEGMENTATION)
        v2 = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        #output_pred_image = os.path.join("/tmp","debug_image.png")
        #plt.imsave(output_pred_image, v2.get_image()[:, :, ::-1])
        #print("Predicted image saved to: {}".format(output_pred_image))
    else:
        v2 = None
    return outputs, v2


def get_keypoints(outputs, key_num=2, bin_num=10):
    keypoints = outputs["instances"].pred_keypoints
    boxes = outputs["instances"].pred_boxes

    if len(keypoints) > 0:
        twists = []
        for num, keypoint in enumerate(keypoints):
            if num > bin_num:
                break
            twist = Twist()
            twist.linear.x = keypoint[key_num][0]
            twist.linear.y = keypoint[key_num][1]
            twist.linear.z = keypoint[key_num][2]
            twists.append(twist)
        return twists
    else:
        return None


def fast_iou_matrix02(prev_boxes, now_boxes):
    # Compute the IoU matrix between the previous and current boxes
    iou_matrix = np.zeros((prev_boxes.shape[0], now_boxes.shape[0]))
    for i in range(prev_boxes.shape[0]):
        for j in range(now_boxes.shape[0]):
            iou_matrix[i, j] = iou(prev_boxes[i], now_boxes[j])
    return iou_matrix


def fast_iou_matrix01(pre_box, now_box, dist_thresh=500):
    iou_matrix = np.zeros((pre_box.shape[0], now_box.shape[0]))
    for i in range(pre_box.shape[0]):
        for j in range(now_box.shape[0]):
            dist = np.sqrt(((pre_box[i][0]+pre_box[i][2])/2 - (now_box[j][0]+now_box[j][2])/2)**2 +
                           ((pre_box[i][1]+pre_box[i][3])/2 - (now_box[j][1]+now_box[j][3])/2)**2)
            if dist > dist_thresh:
                continue
            iou_matrix[i][j] = iou(pre_box[i], now_box[j])
    return iou_matrix


def send_image(im, pub):
    global bridge
    image_message = bridge.cv2_to_imgmsg(im, encoding="bgr8")
    pub.publish(image_message)


def send_as_twist(matched_scores, matched_keys, send_image=False):
    global pub_sp2, pub_sp3
    global twist
    # if matched_boxes and other value sieze is not the same, it is error
    twists = []
    i = 0
    for i in range(matched_keys.shape[0]):
        twist = Twist()
        print(matched_keys.shape)
        try:
            twist.linear.x = matched_keys[i][1][2][0] #* float(SCALE) 
            twist.linear.y = matched_keys[i][1][2][1] #* float(SCALE)
        except:
            twist.linear.x = matched_keys[0][2][0] #* float(SCALE)
            twist.linear.y = matched_keys[0][2][1] #* float(SCALE)

        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = 0
        
        twists.append(twist)

    if i > 0:
        pub.publish(twists[0])
    elif i > 1:
        pub_sp2.publish(twists[1])
    elif i > 2:
        pub_sp3.publish(twists[2])

    if send_image:
        pass
        # # image publish
#        cv_image_dst = v.get_image()[:, :, ::-1]
#        msg_i_ = bridge.cv2_to_imgmsg(cv_image_dst, encoding="bgr8")
#        pub_i.publish(msg_i_)


def callback_phase_info(message):
    global phase_info
    phase_info = message.data
    print("Phase info: {}".format(phase_info))

    # get angular information form messege (Twist)
    phase_info = int(message.angular.z)
    if phase_info <= 100:
        phase_info = 0
    else:
        phase_info = 1


def callback_ros(message):
    global busy
    
    if busy>0:
        busy -= 1
    else:
        callback_ros_core(message)
        busy = 3


def callback_ros_core(message):
    global spc
    global pub
    global pub_i
    global twist
    global pre_boxes
    global predictor
    global count
    global busy
    busy = 0
    print(message.header.frame_id + " : " + str(message.header.seq) + " : " + str(message.header.stamp))
    im = bridge.imgmsg_to_cv2(message, "bgr8")

    # Hangarian Algorithm
    matched_boxes, matched_keys, matched_scores, matched_velocities =\
        test_callback_ros(im, fps=DEFAULT_FPS, METRIC="iou", hybrid_lambda = 0.7)

    # select best candidate
    matched_boxes, matched_keys, matched_scores, matched_velocities =\
        select_best_candidate(matched_boxes, matched_keys, matched_scores, matched_velocities, metric='score', top_k=[5,4])

    # visualize and publish image
    image = visualize_candidate(im, matched_boxes, matched_keys, matched_scores, matched_velocities, 'predictions', count, score_th=0.1)
    send_image(image, pub_i)

    #send keypoints as twist
    send_as_twist(matched_scores, matched_keys, send_image=True)


    # display twist value
    print("twist: {}".format(twist))
    busy = False
    
    # outputs = predictor(im)
    # v = SPVisualizer(im[:, :, ::-1],
    #                  metadata=MetadataCatalog.get(DATA_SET_NAME),
    #                  scale=1.5,
    #                  instance_mode=ColorMode.SEGMENTATION)
    # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))


def init_keypoints(DATA_SET_NAME,
                   fullpath_coco_json=None,
                   fullpath_coco_images=None,
                   fullpath_coco_train_json=None,
                   fullpath_coco_train_images=None,
                   config_file=TRAIN_WEIGHT_CONF,
                   output_dir="./output",
                   mode="train"):

    # check config_file is valid
    config_file = os.path.join('./configs', config_file)
    assert os.path.exists(config_file), \
              "*** ArgumentCheckErorr Config file not found at {}".format(config_file)

    if not os.path.exists(output_dir):
        print("Creating output directory: {}".format(output_dir))
        os.makedirs(output_dir)

    if mode == "train":
        DatasetCatalog.clear()

        if not fullpath_coco_train_json:
            assert os.path.exists(fullpath_coco_train_json), \
            "*** ArgumentCheckErorr Annotations not found at {}".format(fullpath_coco_train_json)

        if not fullpath_coco_train_images:
            assert os.path.exists(fullpath_coco_train_images), \
            "*** ArgumentCheckErorr Annotations not found at {}".format(fullpath_coco_train_images)
        register_dataset(DATA_SET_NAME, fullpath_coco_train_json, fullpath_coco_train_images)

    # classes = MetadataCatalog.get("sp09").thing_classes = ["sperm_head","Sperm"]
    # MetadataCatalog.get("sp09").thing_classes = ["sperm_head","Sperm"]

    keypoint_names = ['head', 'neck', 'target', 'tail']
    keypoint_connection_rules = [ ('head', 'neck'), ('neck', 'target'), ('target', 'tail')]
    MetadataCatalog.get(DATA_SET_NAME).thing_dataset_id_to_contiguous_id = {1: 0, 2: 1}
    MetadataCatalog.get(DATA_SET_NAME).keypoint_names = keypoint_names
    MetadataCatalog.get(DATA_SET_NAME).keypoint_flip_map = []
    MetadataCatalog.get(DATA_SET_NAME).keypoint_connection_rules = keypoint_connection_rules
    MetadataCatalog.get(DATA_SET_NAME).evaluator_type = "coco"
    # dataset_dicts = DatasetCatalog.get(DATA_SET_NAME)

    # import default config
    cfg = get_cfg()

    # cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.load_yaml_with_base(config_file)
    # ---------------------------------------------------------------------------- #
    # Keypoint Head
    # ---------------------------------------------------------------------------- #
    cfg.OUTPUT_DIR = output_dir
    return cfg


def template_matching(image, template_image='template.png' , method=cv2.TM_CCORR_NORMED):
    global pub2
    global twist_n

    # convert image to gray if RGB
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # load template image
    template_image = cv2.imread(template_image, 0)

    res = cv2.matchTemplate(gray, template_image, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    xy = max_loc
    # avg = max_loc
    twist_n.linear.x = max_loc[0]
    twist_n.linear.y = max_loc[1]

    pub2.publish(twist_n)

def cv2_imshow(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(25, 8)), plt.imshow(im), plt.axis('off')


def select_best_candidate(matched_boxes, matched_keys, matched_scores, matched_velocities, metric='score', top_k=[10,10]):
    # top_k = [5,2] which means selecting top 5 high score in 1st stage then selecting top 2 high-velocities ones in 2nd stage

    if metric == 'score':
        # sort by score
        if len(matched_boxes) == 0:
            return  matched_boxes, matched_keys, matched_scores, matched_velocities
        else:
            if matched_scores.shape[0] < top_k[1]:
                return  matched_boxes, matched_keys, matched_scores, matched_velocities
            else:
                arr = matched_scores.squeeze()
            try:
                sort_idx = np.argsort(-arr[:, 1])
            except:
                sort_idx = 0

            matched_boxes = np.array(matched_boxes)[sort_idx]
            matched_keys = np.array(matched_keys)[sort_idx]
            matched_scores = np.array(matched_scores)[sort_idx]
            matched_velocities = np.array(matched_velocities)[sort_idx]

        # keep the top 5
        if len(matched_boxes) > top_k[0]:
            matched_boxes = matched_boxes[:top_k[0]]
            matched_keys = matched_keys[:top_k[0]]
            matched_scores = matched_scores[:top_k[0]]
            matched_velocities = matched_velocities[:top_k[0]]
        else:
            matched_boxes = matched_boxes
            matched_keys = matched_keys
            matched_scores = matched_scores
            matched_velocities = matched_velocities

        # sort by velocity
        arr = np.argsort(matched_velocities)
        try:
            sort_idx = np.argsort(-arr[:, 1])
            matched_boxes = np.array(matched_boxes)[sort_idx]
            matched_keys = np.array(matched_keys)[sort_idx]
            matched_scores = np.array(matched_scores)[sort_idx]
            matched_velocities = np.array(matched_velocities)[sort_idx]

            # remove too learge velocity > 500
            matched_boxes = matched_boxes[matched_velocities[:, 1] < 120]
            matched_keys = matched_keys[matched_velocities[:, 1] < 120]
            matched_scores = matched_scores[matched_velocities[:, 1] < 120]
            matched_velocities = matched_velocities[matched_velocities[:, 1] < 120]

        except:
            sort_idx = 0
            matched_boxes = np.array(matched_boxes)
            matched_keys = np.array(matched_keys)
            matched_scores = np.array(matched_scores)
            matched_velocities = np.array(matched_velocities)

            # remove too learge velocity > 500
            matched_boxes = matched_boxes[matched_velocities< 120]
            matched_keys = matched_keys[matched_velocities < 120]
            matched_scores = matched_scores[matched_velocities< 120]
            matched_velocities = matched_velocities[matched_velocities < 120]

        # keep the top 2
        if len(matched_boxes) > top_k[1]:
            matched_boxes = matched_boxes[:top_k[1]]
            matched_keys = matched_keys[:top_k[1]]
            matched_scores = matched_scores[:top_k[1]]
            matched_velocities = matched_velocities[:top_k[1]]
        else:
            matched_boxes = matched_boxes
            matched_keys = matched_keys
            matched_scores = matched_scores
            matched_velocities = matched_velocities

    return  matched_boxes, matched_keys, matched_scores, matched_velocities


def visualize_candidate(im, matched_boxes, matched_keys, matched_scores, matched_velocities, output_dir, frame_id, score_th=0.1):
    # visualize the matched boxes

    height, width = im.shape[:2]
    center = (width // 2, height // 2)

    for i in range(matched_boxes.shape[0]):
        if  len(matched_boxes[i]) == 2:
            box = matched_boxes[i][1]
            key = matched_keys[i][1]
        else:
            box = matched_boxes[i]
            key = matched_keys[i]

        try:
            score = matched_scores[i][1]
        except:
            if len(matched_scores) > 0:
                score = matched_scores[0]
            else:
                score = matched_scores[0]
        try:
            velocity = matched_velocities[i][1]
        except:
            velocity = matched_velocities[0]

        if score < score_th:
            continue

        #round up the score 2 decimal
        score = score.round(2)
        velocity = velocity.round(2)

        # if rect is out of image, skip it
        if int(box[0]) < 0 or int(box[1]) < 0 or int(box[2]) > width or int(box[3]) > height:
            continue
        else:
            if int(box[1]) > 120:
                sub = 120
            else:
                sub = 0
            cv2.putText(im, "#"+str(i + 1),        (int(box[0]), int(box[1]) - sub),      cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.putText(im, "Score: " + str(score), (int(box[0]), int(box[1]) + 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.putText(im, "vel: " + str(velocity), (int(box[0]), int(box[1]) + 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        points = []
        for j in range(len(key)):
            cv2.circle(im, (int(key[j][0]), int(key[j][1])), 20, (255, 0, 0), -1)
            cv2.line(im, (int(key[0][0]), int(key[0][1])),  (int(key[1][0]), int(key[1][1])), (0, 0, 255), 2)
            cv2.line(im, (int(key[1][0]), int(key[1][1])),  (int(key[2][0]), int(key[2][1])), (0, 0, 255), 2)
            cv2.line(im, (int(key[2][0]), int(key[2][1])),  (int(key[3][0]), int(key[3][1])), (0, 0, 255), 2)
            points.append((int(key[j][0]), int(key[j][1])))

# Add some padding to the rectangle size
        rect = cv2.minAreaRect(np.array(points))

        padding = 200  # adjust this value as needed
        rect = (rect[0], (rect[1][0] + padding, rect[1][1] + padding), rect[2])

        # Draw the rectangle on the image
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(im, [box], 0, (0, 255, 0), thickness=10)

        # matched_boxes = np.array(matched_boxes)
        # matched_keys = np.array(matched_keys)
        # matched_scores = np.array(matched_scores)
        # matched_velocities = np.array(matched_velocities)
        # matched_boxes_center = (matched_boxes[:, 1, 0:2] + matched_boxes[:, 1, 2:4]) / 2
        # matched_boxes_center = matched_boxes_center.astype(np.int)
        # matched_boxes_center_distance = np.sqrt(np.sum(np.square(matched_boxes_center - np.array([960, 540])), axis=1))

    # resize the image to 640x480
    im = cv2.resize(im, (640, 480))
    # save the image
    cv2.imwrite(os.path.join(output_dir, 'frame_{}.png'.format(frame_id)), im)
    return im


def test_callback_ros(im, fps=DEFAULT_FPS, METRIC="hybrid", hybrid_lambda = 0.7):
    global pre_boxes
    global pre_keys
    global pre_scores
    # Threshold of Life frame
    linger_length = 4

    outputs, v = core_predict(im, MetadataCatalog)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #plt.figure(figsize=(25, 8)), plt.imshow(im), plt.axis('off')
    v = v.get_image()
    #cv2_imshow(v.get_image()[:, :, ::-1])
    v = cv2.cvtColor(v, cv2.COLOR_BGR2RGB)
    #plt.figure(figsize=(10, 4)), plt.imshow(v), plt.axis('off')
    boxes = np.array(outputs["instances"].pred_boxes.tensor.cpu())

    keypoints = np.array(outputs["instances"].pred_keypoints.cpu().numpy())
    scores = np.array(outputs["instances"].scores.cpu().numpy()).reshape(-1, 1)
    # chage (n,) -> (n,1)

    # black_list of empty matrix
    black_list = []

    # if pre_boxes is empty, initialize

    if pre_boxes == []:
        pre_boxes = boxes.copy()
        #print(pre_boxes.shape)
        pre_keys = keypoints.copy()
        #print(pre_keys.shape)
        pre_scores = scores.copy()
        return boxes, keypoints, scores, np.array([0])
    else:
        now_boxes = boxes.copy()
        now_keys = keypoints.copy()
        now_scores = scores.copy()
        # convert (8,4,3) -> (8,12)
        #pre_keys_vec = pre_keys.reshape(pre_keys.shape[0],-1)[:,:-1]
        #now_keys_vec = now_keys.reshape(now_keys.shape[0],-1)[:,:-1]
        pre_keys_cp = pre_keys.copy()
        pre_keys_vec = pre_keys_cp.reshape(np.array(pre_keys).shape[0], -1) if np.array(pre_keys).shape[0] != 0 else pre_keys_cp
        #print("pre_boxes: ",pre_boxes.shape)
        #print("pre_keys ",pre_keys.shape)
        if (now_keys.shape[0] == 0) or (now_boxes.shape[0] == 0):
            childless_boxes = pre_boxes.copy()
            if black_list == []:
                black_list = childless_boxes.copy()
            else:
                black_list = np.append(black_list, childless_boxes, axis=0)

            # return None
            pre_boxes = boxes.copy()
            pre_keys = keypoints.copy()
            pre_scores = scores.copy()
        #    print("2 pre_boxes: ",pre_boxes.shape)
        #    print("2 pre_keys: ",pre_keys.shape)    
            return boxes, keypoints, scores, np.array([0])

        else:
            now_keys_vec = now_keys.reshape(now_keys.shape[0], -1)
            dist_matrix = cdist( pre_keys_vec, now_keys_vec, metric='euclidean') if pre_keys_vec.shape[0] != 0 else np.array([])

        # count time for iou_matrix
        start = time.time()
        dist_tred = 20000
        iou_matrix = fast_iou_matrix01(pre_boxes, now_boxes, dist_thresh=dist_tred)
        end = time.time()
        # print("time for iou_matrix: ", end - start)

        # Hungarian algorithm
        # count time for hungarian algorithm
        start = time.time()

        if METRIC == 'iou':
            row_ind, col_ind = linear_sum_assignment(iou_matrix, maximize=True)
        elif METRIC == 'dist':
            row_ind, col_ind2 = linear_sum_assignment(dist_matrix, maximize=False)
        elif METRIC == 'hybrid':
            _lambda = hybrid_lambda
            if (dist_matrix.shape[0] == iou_matrix.shape[0]):
                if (dist_matrix.shape[0] == 0) or (dist_matrix.shape[1] == 0):
                    row_ind = []
                    col_ind = []
                else :
                    row_ind, col_ind = linear_sum_assignment( _lambda*dist_matrix - (1-_lambda)*iou_matrix, maximize=False)
            else:
                row_ind, col_ind = linear_sum_assignment( dist_matrix , maximize=False)

        end = time.time()
        print("time for hungarian algorithm: ", end - start)

        start = time.time()
        matched_boxes = []
        matched_keys = []
        matched_scores = []
        matched_boxes_append = matched_boxes.append
        matched_keys_append = matched_keys.append

        for i, j in zip(row_ind, col_ind):
            dist01 = np.sqrt(np.sum((pre_boxes[i] - now_boxes[j]) ** 2))

            matched_boxes_append((pre_boxes[i], now_boxes[j]))
            matched_keys_append((pre_keys[i], now_keys[j]))
            matched_scores.append((pre_scores[i], now_scores[j]))


        # Get the unmatched boxes in pre_box
        childless_boxes = pre_boxes.copy()
        childless_keys = pre_keys.copy()
        childless_scores = pre_scores.copy()
        orphan_boxes = now_boxes.copy()
        orphan_keys = now_keys.copy()
        orphan_scores = now_scores.copy()

        for match in matched_boxes:
            m_pre_box = match[0]
            m_now_box = match[1]
            for i, pre_box in enumerate(childless_boxes):
                if np.array_equal(m_pre_box, pre_box):
                    # remove the matched box from child_less
                    if childless_boxes.shape[0] != 0:
                        childless_boxes = np.delete(childless_boxes,
                                                    i,
                                                    axis=0)
                    if childless_scores.shape[0] != 0:
                        childless_scores = np.delete(childless_scores,
                                                     i,
                                                     axis=0)
                    # remove the i-th from childless_keys
                    if childless_keys.shape[0] != 0:
                        childless_keys = np.delete(childless_keys,
                                                   i,
                                                   axis=0)
                    # print("4 childless_boxes: ", childless_boxes.shape)
                    # print("4 childless_keys: ", childless_keys.shape)
                    # print("4 childless_scores: ", childless_scores.shape)

                    # using this roop, we get keypoints of the matched boxes
                    break
                else:
                    pass
            for i, now_box in enumerate(orphan_boxes):
                if np.array_equal(m_now_box, now_box):
                    orphan_boxes = np.delete(orphan_boxes, i, axis=0)
                    orphan_scores = np.delete(orphan_scores, i, axis=0)
                    orphan_keys = np.delete(orphan_keys, i, axis=0)
                    # print("4 orphan_boxes: ", orphan_boxes.shape)
                    # print("4 orphan_keys: ", orphan_keys.shape)
                    # print("4 orphan_scores: ", orphan_scores.shape)              
                    break

        end = time.time()
        print("save boxed data into variables ", end - start)

        start = time.time()
        matched_velocities = []
        matched_velocities_append = matched_velocities.append
        time_diff = 1.0 / float(fps)
        for match in matched_boxes:
            diff = match[1] - match[0]
            diff_x = (diff[0] + diff[2]) / 2.
            diff_y = (diff[1] + diff[3]) / 2.
            velocitity = np.array([diff_x / time_diff, diff_y / time_diff])
            matched_velocities_append(velocitity)
        end = time.time()
        print("calc velocities: ", end - start)

        #----------------------------------------------------
        # update pre_boxes and pre_keys and pre_scores
        #----------------------------------------------------
        pre_boxes = boxes.copy()
        pre_keys = keypoints.copy()
        pre_scores = scores.copy()
        # print("3 pre_boxes: ", pre_boxes.shape)
        # print("3 pre_keys: ", pre_keys.shape)
        # print("3 pre_scores: ", pre_scores.shape)

        # ---------------------------------------------------
        # update black_list ( manage life-time of disappeared boxes )
        #----------------------------------------------------
        if black_list == []:
            black_list = childless_boxes.copy()
        else:
            # if box is disappeared, so add it to black_list and delete later from pre_boxes
            #  if it is lingering in black_list for linger_length times
            black_list = np.append(black_list, childless_boxes, axis=0)

        # if disappeared box is listed in black_list multiple times,
        # if the same vector is listed in tol times in black_list, delete it
        for i, box in enumerate(black_list):
            if np.sum(black_list == box, axis=0).all() >= linger_length:
                black_list = np.delete(black_list, i, axis=0)
                print("delete box from black_list: ", box)
                # remove box from pre_boxes
                for j, pre_box in enumerate(pre_boxes):
                    if np.array_equal(pre_box, box):
                        pre_boxes = np.delete(pre_boxes, j, axis=0)
                        pre_keys = np.delete(pre_keys, j, axis=0)
                        pre_scores = np.delete(pre_scores, j, axis=0)
                        break
        # print("4 pre_boxes: ", pre_boxes.shape)
        # print("4 pre_keys: ", pre_keys.shape)
        # print("4 pre_scores: ", pre_scores.shape)

        pre_boxes = np.append(pre_boxes, childless_boxes, axis=0)
        pre_boxes = np.append(pre_boxes, orphan_boxes, axis=0)
        pre_keys = np.append(pre_keys, childless_keys, axis=0)
        pre_keys = np.append(pre_keys, orphan_keys, axis=0)
        pre_scores = np.append(pre_scores, childless_scores, axis=0)
        pre_scores = np.append(pre_scores, orphan_scores, axis=0)
        # print("5 pre_boxes: ", pre_boxes.shape)
        # print("5 pre_keys: ", pre_keys.shape)
        # print("5 pre_scores: ", pre_scores.shape)
    # print shape of all retunred variables
    print("matched_boxes: ", np.array(matched_boxes).shape)
    print("matched_keys: ", np.array(matched_keys).shape)
    print("scores: ", np.array(scores).shape)
    print("matched_velocities: ", np.array(matched_velocities).shape)

    # abs of velocity
    matched_velocities = np.abs(matched_velocities)
    # sort matched_boxes by velocity
    matched_boxes = np.array(matched_boxes)
    matched_keys = np.array(matched_keys)
    matched_scores = np.array(matched_scores)
    scores = np.array(scores)

    if matched_velocities.shape[0] == 0:
        return matched_boxes, matched_keys, matched_scores.squeeze(), matched_velocities
    else:
        matched_boxes = matched_boxes[np.argsort(matched_velocities[:, 0])]
        matched_keys = matched_keys[np.argsort(matched_velocities[:, 0])]
        matched_scores = matched_scores[np.argsort(matched_velocities[:, 0])]
        return matched_boxes, matched_keys, matched_scores.squeeze(), matched_velocities


def predict(cfg,
            image_path,
            model_weight,
            config_file=DETECT_config1,
            sample=10,
            output_dir="pred",
            device="cuda",
            video="tst.avi",
            sub_image_node="/sperm_test_image/image_raw",
            sub_twist_node="/sperm_phase_switch",
            mode = "ROS",
            fps = DEFAULT_FPS
            ):
    global sub
    global pub
    global predictor
    global pre_boxes
    global busy
    global phase_info   # waitng if attack mode enabled.
    
    #assert os.path.exists(image_path), "Image not found at {}".format(image_path)
    # assert os.path.exists(config_file), "Config file not found at {}".format(config_file)

    if model_weight is None:
        model_weight = os.path.join(BASE_DIR, cfg.OUTPUT_DIR, "model_final.pth")
    assert os.path.exists(model_weight), "Model weight not found at {}".format(model_weight)

    # output dir
    os.makedirs(output_dir, exist_ok=True)

    # defaut config file
    # cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.merge_from_file(os.path.join('configs', config_file))
    # override config by args
    cfg.MODEL.WEIGHTS = model_weight
    cfg.MODEL.KEYPOINT_ON = True
    # cfg.MODEL.DEVICE = device
    # use cpu not gpu
    #cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.DEVICE = device
    cfg.TEST.KEYPOINT_OKS_SIGMAS = np.array([1,2,2,2], dtype=float).tolist()
    cfg.OUTPUT_DIR = output_dir
    predictor = DefaultPredictor(cfg)
    cfg.DATALOADER.NUM_WORKERS = 0

    if mode == "ROS":
        ROS = True
        VIDEO = False

    if mode == "VIDEO":
        VIDEO = "tst.avi"
        ROS = False

    if ROS:
        init_ros()
        busy = False
        # r = rospy.Rate(fps)
        sub = rospy.Subscriber(sub_image_node, Image, callback_ros)
        phase_info_sub = rospy.Subscriber(sub_twist_node, 
                                                                    Twist,
                                                                    callback_phase_info)
        # r.sleep()
        rospy.spin()
    else:
        if VIDEO:
            cap = cv2.VideoCapture(VIDEO)
            while cap.isOpened():
                ret, im = cap.read()

                im = cv2.resize(im, (im.shape[0]//SCALE, im.shape[1]//SCALE))

                outputs, v = core_predict(im, MetadataCatalog)
                cv2_imshow(v.get_image()[:, :, ::-1])
                keypoints = outputs["instances"].pred_keypoints
                if keypoints is not None:
                    # print(keypoints)
                    print(keypoints.shape)
        else:
            # open image file in the directory
            image_dir = "./20230204keypoints/images/"
            import glob
            # loop through all the images of png files in the image_dir
            for i, image_path in enumerate(glob.glob(os.path.join(image_dir, "*.png"))):

                im = cv2.imread(image_path)

                if i == 3:
                    print('debug')
                a, b, c, d = test_callback_ros(im)

                if (d.shape[0] == 1) and d == 0:
                    print("no previous frame or no sperm detected")
                else:
                    a, b, c, d = select_best_candidate(a, b, c, d, metric='score')
                    image = visualize_candidate(im, a, b, c, d, 'predictions',  i)
                    #send image

                    if not DEBUG_MODE:
                        send_image(image, pub_i)
                        #send keypoints as twist
                        send_as_twist(c, b, send_image=False)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, default=BASE_DIR, help='base dir')
    parser.add_argument('--dataset', type=str, default=None, help='this option is deprecated')
    parser.add_argument('--interactive', type=bool, default=True, help='interactive mode')
    parser.add_argument('--weight', type=str, default='model_final.pth', help='model weight')
    parser.add_argument('--output', type=str, default='output', help='output dir')
    parser.add_argument('--source', type=str, default="20230204keypoints/images", help='input source dir (test imaeges dir)')
    parser.add_argument('--sample', type=int, default=0, help='sample number')
    parser.add_argument('--config', type=str, default=TRAIN_WEIGHT_CONF, help='config file')
    parser.add_argument('--label', type=str, default=LABEL_DIR, help='label name relative to base dir')
    parser.add_argument('--image', type=str, default="annotations/images", help='training images dir')
    parser.add_argument('--device', type=str, default="cuda", help='device')
    #parser.add_argument('--sub_image_node', type=str, default="sperm_test_image/image_raw")
    parser.add_argument('--sub_image_node', type=str, default="/stcamera_node/dev_142122B11642/image_raw")
    # parser.add_argument('--sub_image_node', type=str, default="/image_data")
    #parser.add_argument('--sub_image_node', type=str, default="sperm_test_image")
    parser.add_argument('--mode', type=str, default="ROS", help="VIDEO ROS IMAGE")
    parser.add_argument('--fps', type=int, default=DEFAULT_FPS, help="frame per second")
    args = parser.parse_args()
    mode = args.mode
    BASE_DIR = args.basedir
    DEVICE = args.device
    if args.dataset:
        DATA_SET_NAME = args.dataset

    FPS = args.fps
    INTERACTIVE = args.interactive
    WEIGHT = args.weight
    OUTPUT_DIR = args.output
    SOURCE_DIR = args.source
    SAMPLES = args.sample
    LABEL = args.label
    IMAGE = args.image
    DEFAULT_WEIGHT = os.path.join(OUTPUT_DIR, args.weight)
    SUB_IMAGE_NODE = args.sub_image_node
    if not os.path.exists(DEFAULT_WEIGHT):
        print("Default weight file is not found:", DEFAULT_WEIGHT)
        exit(1)

    cfg = init_keypoints(DATA_SET_NAME, mode="detect")
    image_path = SOURCE_DIR

    predict(cfg,
            image_path,
            model_weight=DEFAULT_WEIGHT,
            config_file=DETECT_config2,
            sample=SAMPLES,
            output_dir=OUTPUT_DIR,
            device=DEVICE,
            sub_image_node=SUB_IMAGE_NODE,
            mode=mode,
            fps=FPS
            )
