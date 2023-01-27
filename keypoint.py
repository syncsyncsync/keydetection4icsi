#!/usr/bin/env python
# coding: utf-8
#
# Keypoint Detection on ICSI Dataset
#
# Akira Okutomi <okutomi@pm.me>
#
# 2020-12-12: Added data augmentation


import os
import glob
import json
import random
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse 
import time 
from enum import Enum
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

K101 = 'ICSI-DETECTION/SP_keypoint_rcnn_R_101_FPN_3x.yaml'

#K50 = "./configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
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
        T.RandomCrop("absolute", (256, 256)),
        T.RandomRotation(angle=[-90, 90])
        ]
        mapper = DatasetMapper(cfg, is_train=True, augmentations = my_augmentations)
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

def init_keypoints(DATA_SET_NAME,
                   fullpath_coco_json=None,
                   fullpath_coco_images=None,
                   config_file=K101,
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
        if fullpath_coco_json:
            assert os.path.exists(fullpath_coco_json), \
            "*** ArgumentCheckErorr Annotations not found at {}".format(fullpath_coco_json)
        if fullpath_coco_images:
            assert os.path.exists(fullpath_coco_images), \
            "*** ArgumentCheckErorr Annotations not found at {}".format(fullpath_coco_images)
        
        register_dataset(DATA_SET_NAME, fullpath_coco_json, fullpath_coco_images)

    # classes = MetadataCatalog.get("sp09").thing_classes = ["sperm_head","Sperm"]
    # MetadataCatalog.get("sp09").thing_classes = ["sperm_head","Sperm"]

    keypoint_names = ['head', 'neck', 'target', 'tail']
    keypoint_connection_rules = [ ('head', 'neck'), ('neck', 'target'), ('target', 'tail')]
    MetadataCatalog.get(DATA_SET_NAME).thing_dataset_id_to_contiguous_id = {1: 0, 2: 1}
    MetadataCatalog.get(DATA_SET_NAME).keypoint_names = keypoint_names
    MetadataCatalog.get(DATA_SET_NAME).keypoint_flip_map = []
    MetadataCatalog.get(DATA_SET_NAME).keypoint_connection_rules = keypoint_connection_rules
    MetadataCatalog.get(DATA_SET_NAME).evaluator_type = "coco"
    dataset_dicts = DatasetCatalog.get(DATA_SET_NAME)

    # import default config
    cfg = get_cfg()
    #cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.load_yaml_with_base(config_file)
    # ---------------------------------------------------------------------------- #
    # Keypoint Head
    # ---------------------------------------------------------------------------- #
    cfg.OUTPUT_DIR = output_dir
    # Directory where output files are written
    # Set seed to negative to fully randomize everything.
    # Detail:
    # Set seed to positive to use a fixed seed. Note that a fixed seed increases
    # reproducibility but does not guarantee fully deterministic behavior.
    # Disabling all parallelism further increases reproducibility.
    #cfg.SEED = -1  

    # Benchmark different cudnn algorithms.
    # If input images have very different sizes, this option will have large overhead for about 10k iterations. 
    # It usually hurts total time, but can benefit for certain models.
    # If input images have the same or similar sizes, benchmark is often helpful.
    #cfg.CUDNN_BENCHMARK = False

    # The period (in terms of steps) for minibatch visualization at train time.
    # Set to 0 to disable.
    #cfg.VIS_PERIOD = 0

    # ---------------------------------------------------------------------------- #
    # Keypoint Head
    # ---------------------------------------------------------------------------- #
    #cfg.MODEL.ROI_KEYPOINT_HEAD.NAME = "KRCNNConvDeconvUpsampleHead"
    #cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = 14
    #cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO = 0
    #cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS = tuple(512 for _ in range(8))
    #cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 4
    
    # Images with too few (or no) keypoints are excluded from training.
    #cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE = 1 

    # This should be a list[float] with the same length as ROI_KEYPOINT_HEAD.NUM_KEYPOINTS, in our case 4.
    #cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((4, 1), dtype=float).tolist()

    # Type of pooling operation applied to the incoming feature map for each RoI
    #cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE = "ROIAlignV2"

    # Normalize by the total number of visible keypoints in the minibatch if True.
    # Otherwise, normalize by the total number of keypoints that could ever exist
    # in the minibatch.
    # The keypoint softmax loss is only calculated on visible keypoints.
    # Since the number of visible keypoints can vary significantly between
    # minibatches, this has the effect of up-weighting the importance of
    # minibatches with few visible keypoints. (Imagine the extreme case of
    # only one visible keypoint versus N: in the case of N, each one
    # contributes 1/N to the gradient compared to the single keypoint
    # determining the gradient direction). Instead, we can normalize the
    # loss by the total number of keypoints, if it were the case that all
    # keypoints were visible in a full minibatch. (Returning to the example,
    # this means that the one visible keypoint contributes as much as each
    # of the N keypoints.)
    #cfg.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS = True
    # Multi-task loss weight to use for keypoints
    # Recommended values:
    #   - use 1.0 if NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS is True
    #   - use 4.0 if NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS is False
    #cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT = 1.0

    return cfg

# To verify the data loading is correct
def check_data(DATA_SET_NAME, dataset_dicts, num=5):
    for d in random.sample(dataset_dicts, num):
        img = cv2.imread(d["file_name"])
        visualizer = SPVisualizer(img[:, :, ::-1],
                                metadata=MetadataCatalog.get(DATA_SET_NAME),
                                scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        im = vis.get_image()[:, :, ::-1]
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        plt.figure()
        plt.imshow(im)
        plt.axis('off')


def cv2_imshow(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(25, 8)), plt.imshow(im), plt.axis('off')

def ressume_train(DATA_SET_NAME, cfg, model_weight, config_file=K101):
    print("model_weight:", model_weight)
    print("config_file:", config_file)

    assert os.path.exists(model_weight), "Model weight not found at {}".format(model_weight)
    assert os.path.exists(config_file), "Config file not found at {}".format(config_file)

    train(model_weight,cfg, resume=True ,config_file=K101)
    return cfg


def train(cfg, train_dataset=None, test_dataset=None ,resume=False, model_weight=None ,config_file=K101):
    # args:
    #
    # config_file: config file  for keypoints detection
    # default = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
    #
    # model_weight: model weight file to start from
    # model_weight = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    #cfg.merge_from_file(model_zoo.get_config_file(config_file))
    
    if model_weight == None:
        if os.path.exists(cfg.MODEL.WEIGHTS):
            model_weight = cfg.MODEL.WEIGHTS
        else:
            # print error and exit 
            assert os.path.exists(cfg.MODEL.WEIGHTS), "Model weight not found at {}".format(cfg.MODEL.WEIGHTS)

    else:        
        if os.path.exists(model_weight):
            cfg.MODEL.WEIGHTS = model_weight
        else:
            print("Model weight not found at {}".format(model_weight))
            # exit error if cfg.MODEL.WEIGHTS is not exist
            assert os.path.exists(model_weight), "Model weight not found at {}".format(model_weight)

    cfg.MODEL.DEVICE = "cuda"

    if train_dataset != None:
        cfg.DATASETS.TRAIN = (train_dataset,)

    if test_dataset != None:
        cfg.DATASETS.TEST = (test_dataset,)
    # cfg.DATASETS.TEST = ("hand_test",)  #Dataset 'hand_test' is empty in my case
    #cfg.DATALOADER.NUM_WORKERS = 12
        
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    #trainer = DefaultTrainer(cfg)    #CocoTrainer(cfg)
    trainer = MyTrainer(cfg)    #CocoTrainer(cfg)
    trainer.resume_or_load(resume)
    trainer.train()
    return cfg
    

def predict(cfg,
            image_path,
            model_weight,
            config_file=K101,
            sample=10,
            output_dir="pred",
            device="cuda"
            ):
    '''
    args:
    image_path: path to the image to predict

    model_weight:
    model weight file to start from
    defaul model_weight is  os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    config_file:
    config file  for keypoints detection

    default = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"

    return:
        outputs: detectron2 outputs
        v: detectron2 visualizer
    '''
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
    cfg.MODEL.DEVICE = device
    cfg.TEST.KEYPOINT_OKS_SIGMAS = np.array([1,2,2,2], dtype=float).tolist()    
    cfg.OUTPUT_DIR = output_dir
    predictor = DefaultPredictor(cfg)

    if sample <= 0:
        sample = 999999
    
    for i, file in enumerate(os.listdir(image_path)):
        if i > sample:
            break

        if file.endswith(".jpg") or file.endswith(".png"):
            image_fullpath = os.path.join(image_path, file)
            print("Predicting image: {}".format(image_fullpath))
            im = cv2.imread(image_fullpath)

            # get prediction time for each image
            start = time.time()
            outputs = predictor(im)
            end = time.time()
            print("Prediction time: {}".format(end - start))
    
            v = SPVisualizer(im[:, :, ::-1],
                           metadata=MetadataCatalog.get(DATA_SET_NAME),
                           scale=1.5,
                           instance_mode=ColorMode.SEGMENTATION)
             
        v2 = v.draw_instance_predictions(outputs["instances"].to("cpu"))        
        #v2 = v.draw_and_connect_keypoints(keypoints)
        #v2 = v.draw_and_connect_predictions(outputs["instances"].to("cpu"))        
        output_pred_image = os.path.join(output_dir, file)
        plt.imsave(output_pred_image, v2.get_image()[:, :, ::-1])
        print("Predicted image saved to: {}".format(output_pred_image))


def on_image(image_path, outputs_dir, predictor):
    im = cv2.imread(image_path)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=MetadataCatalog.get(DATA_SET_NAME),
                   scale=1.5,
                   instance_mode=ColorMode.SEGMENTATION
                   )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite(os.path.join(outputs_dir, os.path.basename(image_path)), v.get_image()[:, :, ::-1])


def start_train():
    BASE_DIR = '/home/ssm-user/detectron2/20230104keypoints'
    LABEL_DIR = "annotations/person_keypoints_default.json"
    IMAGE_DIR = "images"
    DATA_SET_NAME = "sp12"
    #print("Base_dir:", BASE_DIR)
    cfg = init_keypoints(DATA_SET_NAME,
                         fullpath_coco_train_json=os.path.join(BASE_DIR, LABEL_DIR),
                         fullpath_coco_train_images=os.path.join(BASE_DIR, IMAGE_DIR))
    train(DATA_SET_NAME, cfg, resume=True, model_weight="model_final.pth", config_file=K101)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="detect", help='train,resume, detect')
    parser.add_argument('--basedir', type=str, default='/home/ssm-user/detectron2/20230104keypoints', help='base dir')
    parser.add_argument('--dataset', type=str, default='sp12', help='dataset name')
    parser.add_argument('--interactive', type=bool, default=True, help='interactive mode')
    parser.add_argument('--weight', type=str, default='model_final.pth', help='model weight')
    parser.add_argument('--output', type=str, default='output', help='output dir')
    parser.add_argument('--source', type=str, default="20230109keypoints_test/images", help='input source dir (test imaeges dir)')
    parser.add_argument('--sample', type=int, default=0, help='sample number')
    parser.add_argument('--config', type=str, default='COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml', help='config file')
    parser.add_argument('--label', type=str, default="annotations/person_keypoints_default.json", help='label name relative to base dir')
    parser.add_argument('--image', type=str, default="annotations/images", help='training images dir')

    args = parser.parse_args()
    mode = args.mode
    BASE_DIR = args.basedir
    DATA_SET_NAME = args.dataset
    INTERACTIVE = args.interactive
    WEIGHT = args.weight
    OUTPUT_DIR = args.output
    SOURCE_DIR = args.source
    SAMPLES = args.sample
    LABEL = args.label
    IMAGE = args.image
    DEFAULT_WEIGHT = os.path.join(OUTPUT_DIR,  args.weight)

    if not os.path.exists(DEFAULT_WEIGHT):
        print("Default weight file is not found:", DEFAULT_WEIGHT)
        exit(1)

    if mode == "detect": 
        print("Base_dir:", BASE_DIR)

        if not os.path.exists(os.path.join(BASE_DIR, LABEL)):
            print( LABEL, " is not found in ", BASE_DIR)
            exit(1)
        if not os.path.exists(os.path.join(BASE_DIR, "images")):
            print( IMAGE, " is not found in ", BASE_DIR)
            exit(1)

        register_dataset(DATA_SET_NAME, os.path.join(BASE_DIR, LABEL), os.path.join(BASE_DIR, IMAGE))

        cfg = init_keypoints(DATA_SET_NAME,
                             mode="detect")
        
        image_path = SOURCE_DIR

        predict(cfg,
                image_path,
                model_weight=DEFAULT_WEIGHT,
                config_file=K101,
                sample=SAMPLES,
                output_dir=OUTPUT_DIR, device="cpu")

    # train
    elif mode == "train":
        # inquire if you mant to train or not    
        print("Base_dir:", BASE_DIR)

        # if "annotations/person_keypoints_default.json" and "images" are not in BASE_DIR then 
        # stop the program

        if not os.path.exists(os.path.join(BASE_DIR, "annotations/person_keypoints_default.json")):
            print("annotations/person_keypoints_default.json is not found in BASE_DIR: ", BASE_DIR)
            exit(1)

        if not os.path.exists(os.path.join(BASE_DIR, "images")):
            print( "images is not found in BASE_DIR: ", BASE_DIR)
            exit(1)

        cfg = init_keypoints(DATA_SET_NAME,
                             fullpath_coco_train_json=os.path.join(BASE_DIR, "annotations/person_keypoints_default.json"),
                             fullpath_coco_train_images=os.path.join(BASE_DIR,"images"))

        if INTERACTIVE:                             
            print("Do you want to train the model? **Previous results might be gone without backup** (y/n)")                   
            answer = input()
        else:
            answer = "y"
        
        if answer == "y":
            train(DATA_SET_NAME, cfg, resume=False, model_weight=K101, config_file=K101)
        else:
            print("No training is done.")

    elif mode == "resume":
        if INTERACTIVE:                                     
            print("Do you want to resume training of current model? (y/n)")
            answer = input()
        else:
            answer = "y"

        if answer == "y":
            launch(
                start_train,
                4,
                num_machines=1,
                machine_rank=0,
                dist_url='auto'
                )

# classes = o.pred_classes[idxofClass]
# scores = o.scores[idxofClass]
# boxes = o.pred_boxes[idxofClass]
# #masks = o.pred_masks[idxofClass]

#Define new instance and set the new values to new instance.
# Note: detectron2 module provides this method set.
# obj = Instances(image_size=(7, 640))
# obj.set('pred_classes', classes)
# obj.set('scores', scores)
# obj.set('pred_boxes', boxes)
#obj.set('pred_masks', masks)
#import detectron2

