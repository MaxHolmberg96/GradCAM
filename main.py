import tensorflow as tf
from tqdm import trange

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
from models.gradcam import GradCAM
from utils import *
from path import *
from tqdm import tqdm

images_list = get_files(ILSVRC2012VAL_PATH)
boundingbox_list = get_files(ILSVRC2012VAL_BB_PATH)
image_index = 48236


model = VGG19(weights="imagenet", classes=1000)
model.summary()
error = []
classification_error = []
progress_bar = tqdm(range(50000))
for image_index in progress_bar:
    original = tf.expand_dims(
        model.load_image(ILSVRC2012VAL_PATH + images_list[image_index]), 0
    )
    image = model.load_image(ILSVRC2012VAL_PATH + images_list[image_index])
    image = model.preprocess_image(image)
    image = tf.expand_dims(image, 0)
    preds = model.predict(image)
    decoded_preds = model.decode_predictions(preds=preds, top=5)
    gradcam = GradCAM(model.model)
    heatmaps, predictions = get_heatmaps_and_bbs(
        gradcam=gradcam, image=image, class_map=get_map_of_classes(preds, decoded_preds)
    )

    groundtruth = format_ground_truth(
        ILSVRC2012VAL_BB_PATH + boundingbox_list[image_index]
    )
    # print(groundtruth)
    # print(evaluate(predictions=predictions, ground_truths=groundtruth))
    predictions = scale_bbs(original.shape, image.shape, predictions)
    # show_image_with_bbs(original, groundtruth, predictions)
    """heatmap = gradcam.get_heatmap(np.argmax(preds), image).numpy()
    show_contours(
        image,
        heatmap,
        (image.shape[2], image.shape[1]),
        np.max(heatmap) * 0.15,
        np.max(heatmap),
    )
    """
    # print("groundtruth", groundtruth, "predictions", predictions)
    # Calculate the good stuff
    classification_error.append(evaluate_classification(predictions, groundtruth))
    error.append(evaluate(predictions, groundtruth))

    progress_bar.set_description(
        "Mean error: {:02f}, Classification error: {:02f}".format(
            np.mean(error), np.mean(classification_error)
        )
    )
# show_image_with_heatmap(gradcam, image, np.argmax(preds))

print(np.mean(error))
print(np.mean(classification_error))
