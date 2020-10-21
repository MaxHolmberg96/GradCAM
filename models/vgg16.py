import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input


class VGG16:
    def __init__(self, weights, classes):
        self.model = tf.keras.applications.VGG16(
            include_top=True,
            weights=weights,
            classes=classes,
            classifier_activation=None,
        )
        self.shape = (self.model.input_shape[1], self.model.input_shape[2])

    def preprocess_image(self, img):
        # Check if grayscale
        # if image.shape[-1] == 1:
        #     image = tf.image.grayscale_to_rgb(image)
        # return tf.image.resize(image, self.shape)

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x

    def load_image(self, file_path):
        img = image.load_img(file_path, target_size=self.shape)

        # return tf.cast(
        #     tf.image.decode_png(tf.io.read_file(file_path)), dtype=tf.float32
        # )
        return img

    def predict(self, X):
        return tf.nn.softmax(self.model.predict(X)).numpy()

    def decode_predictions(self, preds, top):
        return tf.keras.applications.vgg16.decode_predictions(preds=preds, top=top)

    def summary(self):
        self.model.summary()
