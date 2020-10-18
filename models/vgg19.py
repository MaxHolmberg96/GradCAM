import tensorflow as tf


class VGG19:
    def __init__(self, weights, classes):
        self.model = tf.keras.applications.VGG19(
            include_top=True,
            weights=weights,
            classes=classes,
            classifier_activation=None,
        )
        self.shape = (self.model.input_shape[1], self.model.input_shape[2])

    def preprocess_image(self, image):
        return tf.image.resize(image, self.shape)

    def load_image(self, file_path):
        return tf.cast(
            tf.image.decode_png(tf.io.read_file(file_path)), dtype=tf.float32
        )

    def predict(self, X):
        return tf.nn.softmax(self.model.predict(X)).numpy()

    def decode_predictions(self, preds, top=5):
        return tf.keras.applications.vgg19.decode_predictions(preds=preds, top=top)[0]

    def summary(self):
        self.model.summary()
