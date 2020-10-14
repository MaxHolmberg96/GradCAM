import tensorflow as tf
tf.executing_eagerly()

class GradCAM:

    def __init__(self, model):
        self.model = model

    def get_all_layer_outputs(self, X):
        func = tf.keras.backend.function([self.model.layers[0].input], [l.output for l in self.model.layers[1:]])
        layer_outputs = func([X])
        layer_outputs.insert(0, X) # Add so image is the output of the input :)
        return layer_outputs

    def get_gradients(self, index, input, output_last):
        feature_activations = self.model.get_layer(index=index)(input)
        opt = tf.keras.optimizers.SGD()
        gradient_step = opt.compute_gradients(output_last, feature_activations)
        return gradient_step


    def global_average_pooling(self, gradients):
        print(gradients.shape)