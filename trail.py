from tscv import *
import imgaug.augmenters as iaa
import cv2
from ImageBBOx import ImageBBoxLabelList
import matplotlib.pyplot as plt
import os

# Make sure the directory exists
os.makedirs('saved_images', exist_ok=True)

ibll = ImageBBoxLabelList.from_pascal_voc('./data/trainfront/')
# ibll.show_dist() -> Assuming that this function call was displaying something

#split train test
ibll = ImageBBoxLabelList.merge(ibll, ibll)
train_ibll, test_ibll = ibll.split() # removed show parameter assuming it was used to display something

#preprocess of data
def showboxes(data):
    img = data['img']
    gt = data['bboxes']
    for b in gt:
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))
    data['img'] = img
    data['bboxes'] = gt
    return data

# Image Augmentation
iaa_aug = iaa.Sequential([
    iaa.GaussianBlur((0, 1.0)),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.02 * 255), per_channel=False),
    iaa.Multiply((0.8, 1.25), per_channel=False),
    iaa.Add((0, 30), per_channel=False),
    iaa.LinearContrast((0.8, 1.25), per_channel=False),
    iaa.Affine(scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)}, translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}),
    iaa.PerspectiveTransform(scale=(0, 0.01)),
], random_order=False)

RESIZE_W = 1296
RESIZE_H = 1024
iaa_resize = iaa.Resize({'width': RESIZE_W, 'height': RESIZE_H})

train_tfms = [iaa_resize, iaa_aug]
valid_tfms = [iaa_resize]

# Modified function to save images instead of displaying
def display_images(ibll, num_images=6):
    for i in range(min(num_images, len(ibll.data))):
        img = ibll.data[i]['img']
        plt.imshow(img)
        plt.savefig(f'saved_images/image_{i}.png')  # Save the figure to a file
        plt.close()  # Close the figure to free up memory

# usage:
ibll.set_tfms(train_tfms+[showboxes])
ibll.apply_tfms()  # Add this line
display_images(ibll)
