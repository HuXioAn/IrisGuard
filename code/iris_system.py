import warnings
warnings.filterwarnings("ignore")
import os
import sys
import time
import argparse
import cv2
import numpy as np
import scipy
import scipy.io
from PIL import Image
import yaml

sys.path.append("..")


from segmentation.UNet import UNet

from recognition.IrisRecognition import IrisRecognition

import multiprocessing
from multiprocessing import Pool

def get_cfg(cfg_path):
    cfg = yaml.load(open(cfg_path, 'r'))
    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parser for the Open Source Iris System")
    parser.add_argument("--cfg_path",
                        type=str,
                        default="../cfg/cfg.yaml",
                        help="path of the configuration file")
    args = parser.parse_args()
    cfg = get_cfg(args.cfg_path)
    

    img1 = Image.open("../data/img1.jpg").convert('L')
    img2 = Image.open("../data/img2.jpg").convert('L')

    unet = UNet(cfg)
    ir = IrisRecognition(cfg)
    # time for unet segmentation
    time_unet = []
    # time for iris recognition
    time_ir = []
    

    # Segmentation by CC-Net
    unet_start = time.time()
    mask1, mask2 = unet.get_mask([img1, img2])
    mask1, pupil_xyr1, iris_xyr1 = unet.get_circle(mask1)
    mask2, pupil_xyr2, iris_xyr2 = unet.get_circle(mask2)
    time_unet.append(time.time() - unet_start)
    
    
    # Iris recognition (measure time for matching against a template)
    img1_norm = ir.get_rubbersheet(img1, pupil_xyr1[1], pupil_xyr1[0], pupil_xyr1[2], iris_xyr1[2])
    mask1_norm = ir.get_rubbersheet(mask1, pupil_xyr1[1], pupil_xyr1[0], pupil_xyr1[2], iris_xyr1[2])
    code1 = ir.extract_code(img1_norm) 
    
    ir_start = time.time()
    img2_norm = ir.get_rubbersheet(img2, pupil_xyr2[1], pupil_xyr2[0], pupil_xyr2[2], iris_xyr2[2])
    mask2_norm = ir.get_rubbersheet(mask2, pupil_xyr2[1], pupil_xyr2[0], pupil_xyr2[2], iris_xyr2[2])
    code2 = ir.extract_code(img2_norm)
    ir_score = ir.matchCodes(code1, code2, mask1_norm, mask2_norm)
    time_ir.append(time.time() - ir_start)
    
    
    
    # Running time summary
    print('#######################################################')
    print('UNet Summary')
    print('Avg time per pair =', sum(time_unet))
    print('#######################################################')
    
    
    
