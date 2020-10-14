import tensorflow as tf
from models.vgg16 import VGG16
from models.gradcam import GradCAM

vgg16_model = VGG16(weights="imagenet", classes=1000)
vgg16_model.summary()
image = vgg16_model.load_image("images/YellowLabradorLooking_new.jpg")
image = vgg16_model.preprocess_image(image)
image = tf.expand_dims(image, 0)
preds = vgg16_model.predict(image)

gradcam = GradCAM(vgg16_model.model)
layer_outputs = gradcam.get_all_layer_outputs(image)
print()
gradcam.get_gradients(index=17, input=layer_outputs[16], output_last=layer_outputs[-2])
