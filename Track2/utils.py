import os
import re
import sys

from operator import itemgetter, attrgetter

import cv2
import numpy as np

# from skimage.measure import regionprops, label as regionlabel
from skimage_cp import regionprops, label as regionlabel

from PIL import Image
from pathlib import Path


IMG_SIZE_ORIGIN = 576
IMG_SIZE_ORIGIN_DIV_2 = IMG_SIZE_ORIGIN // 2

IMG_SIZE_SELECT = 120
IMG_SIZE_SELECT_DIV_2 = IMG_SIZE_SELECT // 2

kernel_5 = np.ones((5, 5), np.uint8)


def create_mask(img, thrsh=100, border=10):
    # mask = cv2.medianBlur(img, 3)
    mask = (img > thrsh).astype(np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_5)
    mask = cv2.dilate(mask, kernel_5, iterations=1)

    mask[:border, :] = mask[-border:, :] = 0
    mask[:, :border] = mask[:, -border:] = 0

    return mask


def fix_bbox(img_shape, bbox):
    y_min, x_min, y_max, x_max = bbox

    h_box, w_box = y_max - y_min, x_max - x_min
    h_max, w_max = img_shape

    assert h_box <= h_max and w_box <= w_max

    if y_min < 0:
        y_min, y_max = 0, h_box
    elif y_max > h_max:
        y_min, y_max = h_max - h_box, h_max

    if x_min < 0:
        x_min, x_max = 0, w_box
    elif x_max > w_max:
        x_min, x_max = w_max - w_box, w_max

    return y_min, x_min, y_max, x_max


def find_object_bbox(img, area_thrsh=150):
    img_mask = create_mask(img)
    img_mask = regionlabel(img_mask)

    props = regionprops(img_mask)
    props = max(props, key=attrgetter('area'), default=None)

    if props is not None:
        centroid_y, centroid_x = map(int, props.centroid)
    else:
        centroid_y, centroid_x = IMG_SIZE_ORIGIN_DIV_2, IMG_SIZE_ORIGIN_DIV_2

    y_min, x_min, y_max, x_max = (
        centroid_y - IMG_SIZE_SELECT_DIV_2,
        centroid_x - IMG_SIZE_SELECT_DIV_2,
        centroid_y + IMG_SIZE_SELECT_DIV_2,
        centroid_x + IMG_SIZE_SELECT_DIV_2,
    )

    bbox = x_min, y_min, x_max, y_max
    bbox = fix_bbox(img.shape, bbox)
    return bbox


def process_image(img_path, check_target=True):
    img_name = os.path.basename(img_path)

    if check_target:
        img_name_re = re.search('__CYGNO_\d+_\d+_(\w*)_(\d+)_keV', img_name)
        if img_name_re:
            img_class = MAPPING_CLASS[img_name_re.group(1)]
            img_value = MAPPING_VALUE[int(img_name_re.group(2))]
        else:
            img_class, img_value = None, None
    else:
        img_class, img_value = None, None

    img = np.asarray(Image.open(img_path))
    x_min, y_min, x_max, y_max = find_object_bbox(img)
    img = img[y_min:y_max, x_min:x_max]
    img = cv2.medianBlur(img, ksize=3)

    return img, img_class, img_value, img_name


if __name__ == '__main__':
    path = sys.argv[1]
    print('Try to load image from:', path, file=sys.stderr)

    images_paths = Path(path).rglob('*/*.png')
    images_paths = sorted(map(str, images_paths))
    print('Found:', len(images_paths), 'images', file=sys.stderr)
    
    img, *_, img_name = process_image(images_paths[0])
    print('Image "{}" successfully processed'.format(img_name), file=sys.stderr)
