import cv2
import numpy as np


def crop_image(img):
    h, w = img.shape[:2]
    # Case 1
    if (h, w) == (4032, 1960):
        h_margin = 1341
        w_margin = 305
    
    elif (h, w) == (4000, 1800):
        h_margin = 1325
        w_margin = 225

    elif (h, w) == (1800, 4000):
        h_margin = 225
        w_margin = 1325
    
    else:
        h_margin = 0
        w_margin = 0

    wid = w - w_margin*2
    hgt = h - h_margin*2
    img = img[h_margin:-h_margin, w_margin:-w_margin, :]
    img = np.flip(img, 1)
    img = np.transpose(img, (1, 0, 2))
    
    return img



