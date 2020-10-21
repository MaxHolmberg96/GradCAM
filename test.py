import tensorflow as tf

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
from models.vgg16 import VGG16
from path import *
from dataset import *


def top_k_accuracy(y_true, y_pred, k=1):
    """From: https://github.com/chainer/chainer/issues/606

    Expects both y_true and y_pred to be one-hot encoded.
    """
    argsorted_y = np.argsort(y_pred)[:, -k:]
    return np.any(argsorted_y.T == y_true.argmax(axis=1), axis=0).mean()


n = 50000
chunk_size = 200

save_path = "S:\\DD2412\\"
preprocess_and_save(ILSVRC2012VAL_PATH, ILSVRC2012VAL_BB_PATH, save_path, chunk_size=200)

model = VGG16("imagenet", 1000)
accuracy = []
pb = trange(n // chunk_size)
for i in pb:
    x_val, y_val = load_chunk(save_path, i + 1)
    y_pred = model.predict(x_val)
    accuracy.append(top_k_accuracy(y_val, y_pred, k=1))
    pb.set_description("Mean accuracy {}".format(np.mean(accuracy)))

print(np.mean(accuracy))
