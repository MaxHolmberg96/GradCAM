from gradcam_models.gradcam import gradcam
import tensorflow as tf
import cv2
import numpy as np
import synset_mappings


class vgg16(gradcam):
    def __init__(self, weights, classes):
        super().__init__()
        self.model = tf.keras.applications.VGG16(
            include_top=True,
            weights=weights,
            classes=classes,
            classifier_activation=None,
        )
        self.shape = (self.model.input_shape[1], self.model.input_shape[2])

    def predict(self, X):
        return tf.nn.softmax(self.model.predict(X)).numpy()

    def decode_predictions(self, preds, top):
        return tf.keras.applications.vgg16.decode_predictions(preds=preds, top=top)

    def summary(self):
        self.model.summary()

    def preprocess_input(self, x):
        return tf.keras.applications.vgg16.preprocess_input(x)

    def get_heatmap(self, c, data):
        # Get final convolutional layer
        conv_layer = self._get_last_conv_layer()
        grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [self.model.get_layer(conv_layer).output, self.model.output],
        )

        # Calculate gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(data)
            y_c = predictions[:, c]
        conv_output = conv_outputs[0]
        grads = tape.gradient(y_c, conv_outputs)
        grads = grads[0]

        # Calculate gradcam heatmap
        alphas = tf.reduce_mean(grads, axis=[0, 1])
        linear_combinaton = tf.reduce_sum(alphas * conv_output, axis=-1)
        linear_combinaton = cv2.resize(
            linear_combinaton.numpy(), (data.shape[2], data.shape[1]), interpolation=cv2.INTER_CUBIC
        )

        # Normalize heatmap
        cam = tf.nn.relu(linear_combinaton).numpy()
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
        return cam

    def locate(self, image, decoded_preds, percentage=0.15, original_shape=None):
        if image.ndim != 3:
            raise Exception("Image should be of shape (width, height, channels)")
        image = tf.expand_dims(image, 0)
        boundingboxes = []
        for name, _, _ in decoded_preds:
            heatmap = self.get_heatmap(c=synset_mappings.name_to_index[name]["index"], data=image)
            max_area, max_contour = self._get_max_contour(heatmap, percentage)
            if max_area != -1:
                xmin, ymin, w, h = cv2.boundingRect(max_contour)
                xmax = xmin + w
                ymax = ymin + h
                if original_shape is not None:
                    height, width, _ = original_shape
                    xmin *= original_shape[1] / image.shape[2]
                    xmax *= original_shape[1] / image.shape[2]
                    ymin *= original_shape[0] / image.shape[1]
                    ymax *= original_shape[0] / image.shape[1]

                boundingboxes.append(
                    (
                        name,
                        {
                            "xmin": xmin,
                            "ymin": ymin,
                            "xmax": xmax,
                            "ymax": ymax,
                        },
                    )
                )
            else:
                boundingboxes.append((name, None))
        return boundingboxes
