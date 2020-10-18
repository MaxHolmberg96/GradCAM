import tensorflow as tf
import cv2
import matplotlib.pyplot as plt


class GradCAM:
    def __init__(self, model):
        self.model = model

    def _get_last_conv_layer(self):
        conv_layer = "No convolutional layer found"
        for layer in self.model.layers:
            if "conv" in layer.name:
                conv_layer = layer.name
        return conv_layer

    def get_all_layer_outputs(self, X):
        func = tf.keras.backend.function(
            [self.model.layers[0].input],
            [layer.output for layer in self.model.layers[1:]],
        )
        layer_outputs = func([X])
        layer_outputs.insert(0, X)  # Add so image is the output of the input :)
        return layer_outputs

    def get_gradients(self, c, image):
        conv_layer = self._get_last_conv_layer()
        grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [self.model.get_layer(conv_layer).output, self.model.output],
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            y_c = predictions[:, c]

        conv_output = conv_outputs[0]
        grads = tape.gradient(y_c, conv_outputs)[0]
        return grads, conv_output

    def get_heatmap(self, c, image):
        grads, output = self.get_gradients(c, image)
        alphas = tf.reduce_mean(grads, axis=[0, 1])
        linear_combinaton = tf.reduce_sum(alphas * output, axis=-1)
        return tf.nn.relu(linear_combinaton)

    def show(self, heatmap, image):
        plt.imshow(image[0, ...] / 255)
        if heatmap is not None:
            cam = cv2.resize(
                heatmap, (image.shape[2], image.shape[1]), cv2.INTER_LINEAR
            )
            plt.imshow(cam, cmap="jet", alpha=0.5)
        plt.show()
