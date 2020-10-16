import tensorflow as tf
import numpy as np
import xml.etree.ElementTree as ET
import cv2


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
    box = np.array([ymin / height, xmin / width, ymax / height, xmax / width])
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


def draw_bounding_box_from_heatmap(image, heatmap):
    heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[1]), cv2.INTER_LINEAR)
    threshhold = 0.15 * np.max(heatmap)
    thresh = cv2.threshold(heatmap, threshhold, np.max(heatmap), cv2.THRESH_BINARY)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
