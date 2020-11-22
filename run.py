import tensorflow as tf
import numpy as np
from gradcam_models.vgg16 import vgg16
from gradcam_models.sentencecnn import sentencecnn
import argparse
import dataset
from utils import show_image_with_heatmap
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


parser = argparse.ArgumentParser(description="Parse arguments.")
parser.add_argument("--path", help="Path to image")
parser.add_argument("--output", help="File name for output", required=True)
parser.add_argument("--text", action="store_true", help="Use sentiment cnn")
parser.add_argument("--weights", help="Weights for sentiment cnn")
parser.add_argument("--sentence", help="Sentence to predict sentiment on")
args = parser.parse_args()

if not args.text:
    if not args.path:
        print("Path is required for GradCAM VGG16")
        exit(0)
    model = vgg16("imagenet", classes=1000)
    original_image = dataset.load_original_image(args.path)
    image = dataset.load_vgg_image(args.path, model)
    image = tf.expand_dims(image, 0)
    pred = model.predict(image)
    print(model.decode_predictions(pred, 5))
    c = np.argmax(pred)
    heatmap = model.get_heatmap(c, image)
    show_image_with_heatmap(original_image, heatmap, args.output)

else:
    if not args.weights or not args.sentence:
        print("Weights and sentence is required to run GradCAM with sentence cnn")
        exit(0)
    model = sentencecnn("dataset-text/train.tsv/train.tsv", "dataset-text/glove.6B.200d.txt")
    model.load_data()
    model.load_weights(args.weights)
    x = model.convert_sentence(args.sentence)
    pred = model.predict(tf.expand_dims(x, 0))
    c = np.argmax(pred)
    heatmap1 = model.get_heatmap(c, tf.expand_dims(x, 0), "conv2d")
    heatmap2 = model.get_heatmap(c, tf.expand_dims(x, 0), "conv2d_1")
    heatmap3 = model.get_heatmap(c, tf.expand_dims(x, 0), "conv2d_2")
    heatmap_avg = np.mean(np.stack([heatmap1, heatmap2, heatmap3], axis=-1), -1)

    reverse_dict = {}
    for key in model.sst2_word_index.keys():
        reverse_dict[model.sst2_word_index[key]] = key

    sentence = []
    for i, el in enumerate(x):
        if el != 0:
            sentence.append(el)

    sentence = list(map(lambda y: reverse_dict[y], sentence))
    length = len(sentence)

    heatmap_plot = heatmap_avg.T[:, -length:] / np.max(heatmap_avg.T[:, -length:])
    pad = 20 - length
    heatmap_plot = np.pad(heatmap_plot, [(0, 0), (0, pad)])
    fig = plt.figure(figsize=(50, 4))
    ax = fig.add_subplot(111)
    im = ax.imshow(heatmap_plot)
    plt.xticks(list(range(length)), sentence)
    plt.yticks([])
    ax.set_aspect(1)
    plt.xticks(size=25)
    plt.savefig(args.output, dpi=200, bbox_inches="tight", pad_inches=0)
