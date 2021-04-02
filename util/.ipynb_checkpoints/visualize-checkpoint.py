import albumentations as A
import cv2
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White

def visualize_bbox(img, bbox, w=0, h=0, class_name=None, color=BOX_COLOR, thickness=2, IsNormalize=True):
    """Visualizes a single bounding box on the image"""
    if IsNormalize  :
        x, y, width, height = bbox

        x_min = int((x - width)*w)
        y_min = int((y - height)*h)
        x_max = int((x + width)*w)
        y_max = int((y + height)*h)   
        
    else :
#         print(type(bbox))
#         bbox[:,2:] /= 2
#         bbox[:,:2] += bbox[:,2:] 
        x_min, y_min, x_max, y_max = list(map(int, bbox))
#     print(x_min, y_min, x_max, y_max)

    img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    
#     ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
#     cv2.rectangle(img, (y_min - int(1.3 * text_height), x_min ), (y_min, x_min + text_width ), BOX_COLOR, -1)
#     cv2.putText(
#         img,
#         text=class_name,
#         org=(x_min, y_min - int(0.3 * text_height)),
#         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#         fontScale=0.35, 
#         color=TEXT_COLOR, 
#         lineType=cv2.LINE_AA,
#     )
    return img

def visualize(image, bboxes, w=0, h=0, category_ids=None, category_id_to_name=None, IsNormalize=True ):
    if type(image) is np.ndarray :
        img = image.copy()
    else : # pytorch tensor
        img = image.clone().data.cpu().numpy()
#     for bbox, category_id in zip(bboxes, category_ids):
#         class_name = category_id_to_name[category_id]
#         print(class_name)
    for bbox in bboxes :
#         print(bbox)
        img = visualize_bbox(img, bbox, w, h, class_name = None, IsNormalize=IsNormalize)
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(img)    