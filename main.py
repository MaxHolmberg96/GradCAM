import tensorflow as tf
from models.vgg16 import VGG16

vgg16_model = VGG16(weights="imagenet", classes=1000)
vgg16_model.summary()
image = vgg16_model.load_image("images/YellowLabradorLooking_new.jpg")
image = vgg16_model.preprocess_image(image)
image = tf.expand_dims(image, 0)
preds = vgg16_model.predict(image)
preds_before_softmax = vgg16_model.get_all_layer_outputs(image)
print(preds.shape)
print("----------------------------------------")
print(len(preds_before_softmax))
for layer in preds_before_softmax:
    print(layer.shape)
#print(vgg16_model.decode_predictions(preds))