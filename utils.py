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


def evaluate(predictions, ground_truths):
    min_error_list = []
    for prediction in predictions:
        max_error_list = []
        for ground_truth in ground_truths:
            if prediction[0] == ground_truth[0]:
                d = 0
            else:
                d = 1
            if overlap(prediction[1], ground_truth[1]) > 0.5:
                f = 0
            else:
                f = 1
            max_error_list.append(max(d, f))
        min_error_list.append(min(max_error_list))
    return min(min_error_list)


def overlap(rect1, rect2):
    intersect_area = max(
        0, min(rect1["x_max"], rect2["x_max"]) - max(rect1["x_min"], rect2["x_min"])
    ) * max(
        0, min(rect1["y_max"], rect2["y_max"]) - max(rect1["y_min"], rect2["y_min"])
    )
    area_rect1 = (rect1["x_max"] - rect1["x_min"]) * (rect1["y_max"] - rect1["y_min"])
    area_rect2 = (rect2["x_max"] - rect2["x_min"]) * (rect2["y_max"] - rect2["y_min"])
    return intersect_area / (area_rect1 + area_rect2 - intersect_area)
