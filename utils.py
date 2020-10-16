import tensorflow as tf
import numpy as np
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt


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


def draw_bounding_box_from_heatmap(image, heatmap, max_val):
    heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[1]), cv2.INTER_LINEAR)
    heatmap = np.uint8(heatmap * 255)
    threshold = 0.15 * max_val
    thresh = cv2.threshold(heatmap, threshold, max_val, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    #img = np.uint8(image)
    #cv2.drawContours(img[0], cnts, -1, (0, 255, 0), 3)
    #cv2.imshow('Contours', img[0])
    max_contour = -1
    max_area = -1
    for c in cnts:
        area = cv2.contourArea(c)
        if area > max_area:
            max_area = area
            max_contour = c

    if max_area != -1:
        xmin, ymin, w, h = cv2.boundingRect(max_contour)
        xmax = xmin + w
        ymax = ymin + h
        return draw_bounding_box(image, xmin, ymin, xmax, ymax)
    return image


def get_top_class_indices(preds, top=5):
    return np.argsort(-preds)[0][:top]


def show_image(image):
    plt.imshow(image.numpy()[0, ...] / 255)
    plt.show()

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
