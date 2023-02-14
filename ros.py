#!/usr/bin/env python
# coding: utf-8
# Keypoint Detection on ICSI Dataset
# 2020-12-12: Added data augmentation

# ROS
import rospy
import roslib
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import os
import numpy as np
import cv2
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
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

setup_logger()
K50_1 = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x"
K50_3 = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x"
K101 = "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x"
KX101 = "COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x"

K50_3_path = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
K101_path = "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"
DETECT_config1 = "ICSI-DETECTION/SP_keypoint_rcnn_R_101_FPN_3x.yaml"
DETECT_config2 = "ICSI-DETECTION/SP_keypoint_rcnn_R_50_FPN_3x.yaml"

TRAIN_WEIGHT_CONF = K50_3_path
PRETRAIN_WEIGHT_NAME  = TRAIN_WEIGHT_CONF[:-5]


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
                        self.draw_circle((x, y), color=_BLACK, radius=10)
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
    global pub
    global pub_i
    global pub2
    global pub2_i
    rospy.init_node('sperm_detection2', anonymous=True)

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

    pub2 = rospy.Publisher('needle_twist', Twist, queue_size=10)
    pub2_i = rospy.Publisher('needle_twist_image', Image, queue_size=10)

    bridge = CvBridge()
    spc = 0


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
        print("Predicted image saved to: {}".format(output_pred_image))
    else:
        v2 = None
    return outputs, v2


def get_keypoints(outputs, key_num=2, bin_num=10):
    keypoints = outputs["instances"].pred_keypoints

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


def callback_ros(message):
    global spc
    global pub
    global pub_i
    global twist
    global predictor
    print(message.header.frame_id + " : " + str(message.header.seq) + " : " + str(message.header.stamp))
    im = bridge.imgmsg_to_cv2(message, desired_encoding='passthrough')

    outputs, v = core_predict(im, MetadataCatalog)
    twists = get_keypoints(outputs, key_num=2)

    if twists:
        for twist in twists:
            pub.publish(twist)
        cv_image_dst = v.get_image()[:, :, ::-1]
        msg_i_ = bridge.cv2_to_imgmsg(im, encoding="bgr8")
        pub_i.publish(msg_i_)

    else:
        print('no target found')
    #   Dammy process for image of openCV
    
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


def template_matching(image, template_image='needle.png' , method='cv2.TM_CCOEFF_NORMED'):
    global pub2
    global twist_n

    # convert image to gray if RGB
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray, template_image, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # twist value change using min_loc and max_loc
    avg = (min_loc + max_loc) // 2
    twist.linear.x = avg[0]
    twist.linear.y = avg[1]
    pub2.publish(twist_n)


def cv2_imshow(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(25, 8)), plt.imshow(im), plt.axis('off')


def predict(cfg,
            image_path,
            model_weight,
            config_file=DETECT_config1,
            sample=10,
            output_dir="pred",
            device="cuda",
            video="tst.avi",
            sub_image_node="/sperm_test_image/image_raw",
            mode = "ROS"  # ROS, CAM, VIDEO
            ):
    global sub
    global pub
    global predictor
    assert os.path.exists(image_path), "Image not found at {}".format(image_path)
    # assert os.path.exists(config_file), "Config file not found at {}".format(config_file)

    if model_weight is None:
        model_weight = os.path.join(BASE_DIR, cfg.OUTPUT_DIR, "model_final.pth")
    assert os.path.exists(model_weight), "Model weight not found at {}".format(model_weight)

    # output dir
    os.makedirs(output_dir, exist_ok=True)

    # defaut config file
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    # override config by args
    cfg.MODEL.WEIGHTS = model_weight
    cfg.MODEL.KEYPOINT_ON = True
    cfg.MODEL.DEVICE = device
    cfg.TEST.KEYPOINT_OKS_SIGMAS = np.array([1,2,2,2], dtype=float).tolist()
    cfg.OUTPUT_DIR = output_dir
    predictor = DefaultPredictor(cfg)
    cfg.DATALOADER.NUM_WORKERS = 0

    ROS = False
    if mode == "ROS":
        ROS = True

    VIDEO = None
    if mode == "VIDEO":
        VIDEO = "tst.avi"

    if ROS:
        init_ros()
        r = rospy.Rate(10)
        sub = rospy.Subscriber(sub_image_node, Image, callback_ros)
        r.sleep()
        rospy.spin()
    else:
        if VIDEO:
            cap = cv2.VideoCapture(VIDEO)

            while cap.isOpened():
                ret, im = cap.read()
                outputs, v = core_predict(im, MetadataCatalog)
                cv2_imshow(v.get_image()[:, :, ::-1])
                keypoints = outputs["instances"].pred_keypoints
                if keypoints is not None:
                    # print(keypoints)
                    print(keypoints.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, default=BASE_DIR, help='base dir')
    parser.add_argument('--dataset', type=str, default=None, help='this option is deprecated')
    parser.add_argument('--interactive', type=bool, default=True, help='interactive mode')
    parser.add_argument('--weight', type=str, default='model_final.pth', help='model weight')
    parser.add_argument('--output', type=str, default='output', help='output dir')
    parser.add_argument('--source', type=str, default="20230109keypoints_test/images", help='input source dir (test imaeges dir)')
    parser.add_argument('--sample', type=int, default=0, help='sample number')
    parser.add_argument('--config', type=str, default=TRAIN_WEIGHT_CONF, help='config file')
    parser.add_argument('--label', type=str, default=LABEL_DIR, help='label name relative to base dir')
    parser.add_argument('--image', type=str, default="annotations/images", help='training images dir')
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--sub_image_node', type=str, default="sperm_test_image/image_raw")
    #parser.add_argument('--sub_image_node', type=str, default="sperm_test_image")
    parser.add_argument('--mode', type=str, default="ROS")
    args = parser.parse_args()
    mode = args.mode
    BASE_DIR = args.basedir
    DEVICE = args.device
    if args.dataset:
        DATA_SET_NAME = args.dataset

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
            mode=mode
            )
