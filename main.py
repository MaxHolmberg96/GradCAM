import tensorflow as tf
from tqdm import trange
import dataset

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


from models.vgg19 import VGG19
from models.vgg16 import VGG16
from models.gradcam import GradCAM
from utils import *
from path import *

images_list = get_files(ILSVRC2012VAL_PATH)
boundingbox_list = get_files(ILSVRC2012VAL_BB_PATH)
image_index = 622  # 48236

model = VGG16(weights="imagenet", classes=1000)
model.summary()
gradcam = GradCAM(model.model)
top = 5
n = 50000
batch_size = 200

save_path = "S:\\DD2412\\only_resized\\"
# dataset.preprocess_and_save(ILSVRC2012VAL_PATH, ILSVRC2012VAL_BB_PATH, save_path, chunk_size=200)
localization_error = []
classification_error = []
pb = trange(n // batch_size)
for i in pb:
    # i = 48236 // batch_size
    x_val, _ = dataset.load_chunk(save_path, i + 1)
    y_pred = model.predict(x_val)
    decoded_preds = model.decode_predictions(y_pred, top)
    for j in range(batch_size):
        offset = i * batch_size + j
        original_image = dataset.load_original_image(ILSVRC2012VAL_PATH + images_list[offset])
        original_shape = original_image.shape
        bounding_boxes = gradcam.locate(x_val[j], decoded_preds[j], original_shape=original_shape, percentage=0.15)
        # show_image_with_heatmap(gradcam, tf.expand_dims(x_val[j], 0), c=np.argmax(y_pred[j]))
        groundtruth = dataset.load_ground_truth(ILSVRC2012VAL_BB_PATH + boundingbox_list[offset])
        # show_image_with_bbs(original_image, bounding_boxes[0], groundtruth[0])
        localization_error.append(evaluate(bounding_boxes, groundtruth))
        classification_error.append(evaluate_classification(bounding_boxes, groundtruth))

        pb.set_description(
            "Localization error: {}, Classification error: {}".format(
                np.mean(localization_error), np.mean(classification_error)
            )
        )


print(np.mean(localization_error))
print(np.mean(classification_error))
