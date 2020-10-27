import tensorflow as tf
import os
from models.sentencecnn import SentenceCNN
from models.gradcam import GradCAM
from tqdm import trange

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

cnn = SentenceCNN("dataset-text/train.tsv/train.tsv", "dataset-text/glove.6B.200d.txt")
cnn.load_data()
cnn.initalize_model()
cnn.train("checkpoints_1")
# cnn.load_model("checkpoints_1")

index = 5903
print(cnn.sst2_data["Phrase"].values[index])
print(cnn.sst2_X[index])
c = cnn.sst2_y[index]
print(c)
gradcam = GradCAM(cnn.model)

heatmap = gradcam.get_heatmap(c, tf.expand_dims(cnn.sst2_X[index], 0))
print(heatmap)
# cnn.plot_training_history()
