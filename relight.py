"""
Relight

Detects key subject person and adjusts brightness. 
"""
import cv2
import numpy as np
import os
import sys
import math
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn.config import Config

import argparse


"""Curve table"""
def smoothstep(x, A=255.):
    return A * 0.5 * (np.sin((x/A - 0.5 )* math.pi) + 1)

def adjust_gamma(x, A=255., gamma=1.0):
    invGamma = 1.0 / gamma
    return A * ((x / A) ** invGamma)


"""Mask-RCNN"""

class InferenceConfig(Config):
    # Give the configuration a recognizable name
    NAME = "inference"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # NUMBER OF GPUs to use. For CPU training, use 1
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes


def prepare_model():
    # Load the pre-trained model data
    ROOT_DIR = os.getcwd()
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)
        
    config = InferenceConfig()
    
    model = modellib.MaskRCNN(
        mode="inference", model_dir=MODEL_DIR, config=config
    )
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    return model 

"""Image processing"""

def box_area(box):
    # x, y, w, h
    return box[2] * box[3]
    

def get_target_mask(results):
    # return empty if no results
    if results['rois'].shape[0] == 0:
        return None
    
    masks_ = np.transpose(results['masks'], axes=[2,0,1])
    all_masks = zip(results['class_ids'], results['rois'], masks_)
    person_masks = filter(lambda t: t[0] == 1, all_masks)
    sorted_person_masks = sorted(person_masks, key=lambda t: box_area(t[1]), reverse=True)
    largest = sorted_person_masks[0]
    _, _, target = largest
    return target


def apply_curve(image, mask, curve_table):
    mask_ = np.stack([mask, mask, mask])
    mask_ = np.transpose(mask_, [1,2,0])
    tone_curved = cv2.LUT(image, curve_table)
    return np.where(mask_, tone_curved, image)


def process(image, model):
    results = model.detect([image], verbose=0)
    r = results[0]
    target = get_target_mask(r)

    # gamma=1.3
    curve_table = np.array([adjust_gamma(x, gamma=1.2) for x in np.arange(0,256)]).astype(np.int)
    curved = apply_curve(image, target, curve_table)
    return curved


def pipeline(filepath, target_path, model):
    image = cv2.imread(filepath)
    processed = process(image, model)
    cv2.imwrite(target_path, processed)
    return


def parse_args():
    parser = argparse.ArgumentParser(description="relight")
    parser.add_argument(
        "--src_dir", help="Source dir with [jpg|png] image files",
        default=None, required=True)
    parser.add_argument(
        "--target_dir", help="Target dir with [jpg|png] image files",
        default=None, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    src_dir = args.src_dir
    target_dir = args.target_dir

    model = prepare_model()

    for f in os.listdir(src_dir):
        filepath = os.path.join(src_dir, f)
        outpath = os.path.join(target_dir, f)
        pipeline(filepath, outpath, model)
        print("saved to: ", outpath)

