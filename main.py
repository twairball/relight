from relight.altm_retinex import adaptive_tone_map
from relight.hist_equalize import rgb_equalized, hsv_equalized, clahe
from relight.dynamic_he import dhe
import cv2
import numpy as np
import os
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="relight")
    parser.add_argument(
        "--src_dir", help="Source dir with [jpg|png] image files",
        default=None, required=True)
    parser.add_argument(
        "--target_dir", help="Target dir with [jpg|png] image files",
        default=None, required=True)
    parser.add_argument(
        "--filter", help="Enhancement method, [rgb|hsv|clahe|altm|dhe] default=hsv",
        default=None, required=True)
    return parser.parse_args()

def process(image, filter):

    if filter == 'rgb':
        return rgb_equalized(image)
    elif filter == 'hsv':
        return hsv_equalized(image)
    elif filter == 'clahe':
        return clahe(image)
    elif filter == 'altm':
        return adaptive_tone_map(image)
    elif filter == 'dhe':
        return dhe(image)
    
    return hsv_equalized(image)

def pipeline(filepath, target_path, filter):
    image = cv2.imread(filepath)
    processed = process(image, filter)
    cv2.imwrite(target_path, processed)
    return

if __name__ == '__main__':
    args = parse_args()
    src_dir = args.src_dir
    target_dir = args.target_dir
    filter = args.filter

    for f in os.listdir(src_dir):
        filepath = os.path.join(src_dir, f)
        outpath = os.path.join(target_dir, f)
        pipeline(filepath, outpath, filter)
        print("saved to: ", outpath)

