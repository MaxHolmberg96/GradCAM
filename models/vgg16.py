import tensorflow as tf


class VGG16:

    def __init__(self, weights, classes):
        self.model = tf.keras.applications.VGG16(
            include_top=True,
            weights=weights,
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=classes,
            classifier_activation="softmax",
        )
        self.shape = (224, 224)

    def preprocess_image(self, image):
        return tf.image.resize(image, self.shape)

    def load_image(self, filepath):
        return tf.image.decode_png(tf.io.read_file(filepath))

    def predict(self, X):
        return self.model.predict(X)

    def decode_predictions(self, preds):
        return tf.keras.applications.vgg16.decode_predictions(preds=preds)
