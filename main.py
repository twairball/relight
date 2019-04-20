from relight.altm_retinex import adaptive_tone_map
from relight.hist_equalize import rgb_equalized
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
    return parser.parse_args()

def process(image):
    # adaptive tone map
    tone_mapped = adaptive_tone_map(image)
    # rgb equalized
    equalized = rgb_equalized(tone_mapped)
    return equalized

def pipeline(filepath, target_path):
    image = cv2.imread(filepath)
    processed = process(image)
    cv2.imwrite(target_path, processed)
    return

if __name__ == '__main__':
    args = parse_args()
    src_dir = args.src_dir
    target_dir = args.target_dir

    for f in os.listdir(src_dir):
        filepath = os.path.join(src_dir, f)
        outpath = os.path.join(target_dir, f)
        pipeline(filepath, outpath)
        print("saved to: ", outpath)

