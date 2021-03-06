from PIL import Image
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

#     wid = w - w_margin*2
#     hgt = h - h_margin*2
    img = img[h_margin:-h_margin, w_margin:-w_margin, :]
#     img = np.flip(img, 1)
#     img = np.transpose(img, (1, 0, 2))
    
    return img

def image_crop( img_path):
    print(img_path)
    img = cv2.imread(img_path)
    h, w, ch = img.shape
    h_margin = 0
    w_margin = 0    
    
    if (h, w) == (4032, 1960):
        h_margin = 1341
        w_margin = 305
    elif (h, w) == (4000, 1800) :
        h_margin = 1325
        w_margin = 225
    elif (h, w) == (1800, 4000):
        h_margin = 225
        w_margin = 1325     
    else :
        #how to do in this case??
        pass
        
    img = img[h_margin:-h_margin, w_margin:-w_margin, :]
    cv2.imwrite("target.jpg", img)

def transform_bbox_points(img, bbox_point):
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
    
    xmin, ymin, xmax, ymax = bbox_point
    xmin -= h_margin
    ymin -= w_margin
    xmax -= h_margin
    ymax -= w_margin
    new_bbox_point = [xmin, ymin, xmax, ymax]
    
    return new_bbox_point

def transform_bbox(df):
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
    
    xmin, ymin, xmax, ymax = bbox_point
    xmin -= h_margin
    ymin -= w_margin
    xmax -= h_margin
    ymax -= w_margin
    new_bbox_point = [xmin, ymin, xmax, ymax]
    
    return new_bbox_point


def shift(img, val_x, val_y, points_2d=None, is_normalized=True):
    """ Shift Image and Points
    """
    h, w = img.shape[:2]
    shift_x = int(val_x * w)
    shift_y = int(val_y * h)

    # Get Affine transform matrix
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    
    # For image
    img = cv2.warpAffine(img, M, (w, h))
    
    if points_2d is not None:
        # For 2d points
        points_2d = np.array(points_2d).reshape((-1, 2))
        
        if is_normalized:
            points_2d[:, 0] *= w
            points_2d[:, 1] *= h
        
        ones = np.ones(shape=(len(points_2d), 1))
        points_ones = np.hstack([points_2d, ones])
        transformed_points = np.empty_like(points_2d)
        for idx, point in enumerate(points_ones):
            point = np.dot(M, point)
            transformed_points[idx,] = point
        
        if is_normalized:
            transformed_points[:, 0] /= w
            transformed_points[:, 1] /= h
        return img, transformed_points
    else:
        return img


def rotate(img, value, points_2d=None, is_normalized=True):
    h, w = img.shape[:2]
    if points_2d is not None:
        points_2d = np.array(points_2d).reshape((-1, 2))
        if is_normalized:
            points_2d[:, 0] *= w
            points_2d[:, 1] *= h
        
        c_x = np.mean(points_2d[:, 0])
        c_y = np.mean(points_2d[:, 1])

        # Get Affine transform matrix
        M = cv2.getRotationMatrix2D((c_x, c_y), value, 1.0)

        # For image
        img = cv2.warpAffine(img, M, (w, h))
        # For 2d points
        ones = np.ones(shape=(len(points_2d), 1))
        points_ones = np.hstack([points_2d, ones])
        transformed_points = np.empty_like(points_2d)
        for idx, point in enumerate(points_ones):
            point = np.dot(M, point)
            transformed_points[idx,] = point
        
        if is_normalized:
            transformed_points[:, 0] /= w
            transformed_points[:, 1] /= h
        return img, transformed_points
    
    else:
        c_x = w / 2
        c_y = h / 2
        # Get Affine transform matrix
        M = cv2.getRotationMatrix2D((c_x, c_y), value, 1.0)
        # For image
        img = cv2.warpAffine(img, M, (w, h))
        return img    
    
def draw_rect(img_path, bbox_points, image_label=None, color=(0, 255, 0), thickness=5, is_normalized=False, isCenter=True):
    """ Draw rectangle
    Args:
        img: image
        bbox_points: [xmin, ymin, xmax, ymax]
        color: color rgb value
        thickness: line thickness
        is_normalized: Normalized points or not
    Return:
        img
    """
#     print(bbox_points)
    img = cv2.imread(img_path)
    img = cv2.flip(img, 1)
    h, w = img.shape[:2]
    for cls, box in zip(image_label, bbox_points) :   
        
        if cls == 'Carcinoma' or cls == 'LSIL' or cls == 'HSIL' :
            color = (255, 0, 0)
        elif cls == 'ASCUS' :
            color = (0,255,0)
        else :
            color = (0,0,255)
            
        if is_normalized:
            xmin = int(box[0] * w)
            ymin = int(box[1] * h)
            xmax = int(box[2] * w)
            ymax = int(box[3] * h)
        else:
#             xmin = int(box[0])
#             ymin = int(box[1])
#             xmax = int(box[2])
#             ymax = int(box[3])
            if isCenter :
                xmin, ymin, xmax, ymax = (box[0]-box[2], box[1]-box[3], box[0]+box[2], box[1]+box[3])
            else :
                xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        print(xmin, ymin, xmax, ymax)
#           left high (xmin, ymin) and right low (xmax, ymax)
        print(type(img))
        print(color)
        print(thickness)
        img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, thickness)
#         img = cv2.rectangle(img, (ymin, xmin), (ymax, xmax), color, thickness)
            
    return img

def draw_marks(img, points_2d, color=(0, 255, 0), thickness=3, is_normalized=True):
    if is_normalized:
        h, w = img.shape[:2]
        for (x, y) in points_2d:
            cv2.circle(img, (int(x * w), int(y * h)), 1, color, thickness, 1)
    
    else:
        for (x, y) in points_2d:
            cv2.circle(img, (int(x), int(y)), 1, color, thickness, 1)    