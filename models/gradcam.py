import tensorflow as tf


class GradCAM:
    def __init__(self, model):
        self.model = model

    def get_all_layer_outputs(self, X):
        func = tf.keras.backend.function(
            [self.model.layers[0].input], [layer.output for layer in self.model.layers[1:]]
        )
        layer_outputs = func([X])
        layer_outputs.insert(0, X)  # Add so image is the output of the input :)
        return layer_outputs

    def get_gradients(self, c, conv_layer, image):
        grad_model = tf.keras.models.Model(
            [self.model.inputs], [self.model.get_layer(conv_layer).output, self.model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            loss = predictions[:, c]

        output = conv_outputs[0]
        grads = tape.gradient(loss, conv_outputs)[0]
        return grads, output

    def __call__(self, c, conv_layer, image):
        grads, output = self.get_gradients(c, conv_layer, image)
        alphas = tf.math.reduce_sum(grads, axis=[0, 1])
        result = tf.reduce_sum(alphas * output, axis=-1)
        return tf.nn.relu(result)
