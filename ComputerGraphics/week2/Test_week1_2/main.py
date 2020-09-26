import cv2
import numpy as np

def point_processing(src, type = 'original') :
    dst = np.zeros(src.shape,dtype=np.uint8)