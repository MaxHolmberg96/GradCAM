import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from tqdm import trange
from synset_mappings import *
import tensorflow as tf


def load_original_image(path):
    return cv2.imread(path).astype(np.float32)


def load_vgg_image(path, model):
    img = cv2.imread(path).astype(np.float32)
    img = img[..., ::-1]
    img = model.preprocess_input(img)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    return img


def load_image(path):
    img = cv2.imread(path)
    # Resize
    return tf.image.resize(img, (224, 224), method=tf.image.ResizeMethod.BICUBIC)[:, :, ::-1]


def load_ground_truth(path):
    root = ET.parse(path).getroot()
    objects = root.findall("object")
    ground_truths = []
    for i in range(len(objects)):
        xmin = int(objects[i].find("bndbox").findtext("xmin"))
        ymin = int(objects[i].find("bndbox").findtext("ymin"))
        xmax = int(objects[i].find("bndbox").findtext("xmax"))
        ymax = int(objects[i].find("bndbox").findtext("ymax"))
        cls = objects[i].findtext("name")
        ground_truths.append((cls, {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}))

    return ground_truths


def preprocess_and_save(
    x_input_dir,
    y_input_dir,
    output_dir,
    chunk_size=1000,
):
    x_files = os.listdir(x_input_dir)
    x_files.sort()
    x_files = [x_input_dir + fn for fn in x_files]

    y_files = os.listdir(y_input_dir)
    y_files.sort()
    y_files = [y_input_dir + fn for fn in y_files]

    x_val = np.zeros((chunk_size, 224, 224, 3), dtype=np.float32)
    y_val = np.zeros((chunk_size, 1000))

    for i in trange(len(x_files) + 1):
        if i % chunk_size == 0 and i != 0:
            np.save("{}x_val_{}.npy".format(output_dir, i // chunk_size), x_val)
            np.save("{}y_val_{}.npy".format(output_dir, i // chunk_size), y_val)
            y_val = np.zeros((chunk_size, 1000))

        if i == len(x_files):
            break
        # Load (as BGR)
        img = load_vgg_image(x_files[i])
        # Save (as RGB)
        x_val[i % chunk_size] = img[..., ::-1]  # img[:, :, ::-1]

        # All classes for each image are the same so only has to set for one of the ground truths
        gt = load_ground_truth(y_files[i])
        class_name = gt[0][0]
        y_val[i % chunk_size][name_to_index[class_name]["index"]] = 1


def load_chunk(directory, chunk_index, model):
    x_val = np.load(directory + "x_val_" + str(chunk_index) + ".npy")  # loaded as RGB
    x_val = model.preprocess_input(x_val)  # converted to BGR
    y_val = np.load(directory + "y_val_" + str(chunk_index) + ".npy")
    return x_val, y_val
