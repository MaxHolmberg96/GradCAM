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
    bounding_box = format_ground_truth(file_path)[0][1]
    return draw_bounding_box(
        image,
        bounding_box["xmin"],
        bounding_box["ymin"],
        bounding_box["xmax"],
        bounding_box["ymax"],
    )


def get_contours(heatmap, reshape_size, threshold, max_val):
    heatmap = cv2.resize(heatmap, reshape_size, cv2.INTER_LINEAR)
    heatmap = np.uint8(heatmap * 255)
    thresh = cv2.threshold(
        heatmap, threshold, max_val, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    return cnts


def draw_bounding_box_from_heatmap(image, heatmap, threshold, max_val):
    boundingbox = get_bounding_box_from_heatmap(
        heatmap, (image.shape[2], image.shape[1]), threshold, max_val
    )
    return draw_bounding_box(
        image,
        boundingbox["xmin"],
        boundingbox["ymin"],
        boundingbox["xmax"],
        boundingbox["ymax"],
    )


def scale_bbs(original_shape, image_shape, predictions):
    scale_x = original_shape[2] / image_shape[2]
    scale_y = original_shape[1] / image_shape[1]
    new_predictions = []
    for cls, bb in predictions:
        new_predictions.append(
            (
                cls,
                {
                    "xmin": bb["xmin"] * scale_x,
                    "xmax": bb["xmax"] * scale_x,
                    "ymin": bb["ymin"] * scale_y,
                    "ymax": bb["ymax"] * scale_y,
                },
            )
        )
    return new_predictions


def get_bounding_box_from_heatmap(heatmap, reshape_size, threshold, max_val):
    cnts = get_contours(heatmap, reshape_size, threshold, max_val)
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
        return {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
    return None


def get_top_class_indices(preds, top=5):
    return np.argsort(-preds)[0][:top]


def get_map_of_classes(preds, decoded_preds):
    top_classes = get_top_class_indices(preds, top=len(decoded_preds))
    m = {}
    for cls, name in zip(top_classes, decoded_preds):
        m[cls] = name[0]
    return m


def get_heatmaps_and_bbs(gradcam, image, class_map):
    heatmaps = []
    max_val = 0

    for predicted_class in class_map.keys():
        heatmaps.append(gradcam.get_heatmap(c=predicted_class, image=image).numpy())
        max_val = max(max_val, np.max(np.uint8(heatmaps[-1] * 255)))

    bounding_boxes = []
    for pred_name, heatmap in zip(class_map.values(), heatmaps):
        bounding_boxes.append(
            (
                pred_name,
                get_bounding_box_from_heatmap(
                    heatmap, (image.shape[2], image.shape[1]), 0.15 * max_val, max_val
                ),
            )
        )
    return heatmaps, bounding_boxes


def show_image(image):
    plt.imshow(image.numpy()[0, ...] / 255)
    plt.show()


def show_image_with_bb(image, bb):
    image = draw_bounding_box(image, bb["xmin"], bb["ymin"], bb["xmax"], bb["ymax"])
    plt.imshow(image.numpy()[0, ...] / 255)
    plt.show()


def show_image_with_bbs(image, *bbs_list):
    for bbs in bbs_list:
        for bb in bbs:
            image = draw_bounding_box(
                image, bb[1]["xmin"], bb[1]["ymin"], bb[1]["xmax"], bb[1]["ymax"]
            )
    plt.imshow(image.numpy()[0, ...] / 255)
    plt.show()


def show_image_with_heatmap(gradcam, image, c):
    heatmap = gradcam.get_heatmap(c, image)
    gradcam.show(heatmap.numpy(), image)


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
        0, min(rect1["xmax"], rect2["xmax"]) - max(rect1["xmin"], rect2["xmin"])
    ) * max(0, min(rect1["ymax"], rect2["ymax"]) - max(rect1["ymin"], rect2["ymin"]))
    area_rect1 = (rect1["xmax"] - rect1["xmin"]) * (rect1["ymax"] - rect1["ymin"])
    area_rect2 = (rect2["xmax"] - rect2["xmin"]) * (rect2["ymax"] - rect2["ymin"])
    return intersect_area / (area_rect1 + area_rect2 - intersect_area)


def format_ground_truth(file_path):
    root = ET.parse(file_path).getroot()
    objects = root.findall("object")
    ground_truths = []
    for i in range(len(objects)):
        xmin = int(objects[i].find("bndbox").findtext("xmin"))
        ymin = int(objects[i].find("bndbox").findtext("ymin"))
        xmax = int(objects[i].find("bndbox").findtext("xmax"))
        ymax = int(objects[i].find("bndbox").findtext("ymax"))
        cls = objects[i].findtext("name")
        ground_truths.append(
            (cls, {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax})
        )

    return ground_truths
