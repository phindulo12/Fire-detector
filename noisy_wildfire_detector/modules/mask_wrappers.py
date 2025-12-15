import cv2
import numpy as np


def mask_postprocess(m):
    k = np.ones((5, 5), np.uint8)
    m2 = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m3 = cv2.morphologyEx(m2, cv2.MORPH_OPEN, k)
    return m3
