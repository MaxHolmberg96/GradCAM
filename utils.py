import tensorflow as tf
import numpy as np
import xml.etree.ElementTree as ET

ILSVRC2012VAL_PATH = "S:\\qbittorrent\\ILSVRC2012_img_val\\"
ILSVRC2012VAL_BB_PATH = "S:\\qbittorrent\\ILSVRC2012_bbox_val_v3\\"


def get_files(file_path):
    from os import walk
    f = []
    for _, _, filenames in walk(file_path):
        f.extend(filenames)
    return f

def draw_bounding_box(image, xmin, ymin, xmax, ymax):
    if len(image.shape) == 4:
        height = image.shape[1]
        width = image.shape[2]
    else:
        print("Image need to have shape (batch, height, width, channels)")
        return
    box = np.array([xmin / width, ymin / height, xmax / width, ymax / height])
    boxes = box.reshape([1, 1, 4])
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    return tf.image.draw_bounding_boxes(image, boxes, colors)


def draw_bounding_box_from_file(image, file_path):
    root = ET.parse(file_path).getroot()
    xmin = int(root.find("object").find("bndbox").findtext("xmin"))
    ymin = int(root.find("object").find("bndbox").findtext("ymin"))
    xmax = int(root.find("object").find("bndbox").findtext("xmax"))
    ymax = int(root.find("object").find("bndbox").findtext("ymax"))
    return draw_bounding_box(image, xmin, ymin, xmax, ymax)
