import tensorflow_datasets as tfds
from keras_cv import bounding_box
from tensorflow import data as tf_data
import keras_cv


def unpackage_raw_tfds_inputs(inputs, bounding_box_format):
    image = inputs["image"]
    boxes = keras_cv.bounding_box.convert_format(
        inputs["objects"]["bbox"],
        images=image,
        source="rel_yxyx",
        target=bounding_box_format,
    )
    bounding_boxes = {
        "classes": inputs["objects"]["label"],
        "boxes": boxes,
    }
    return {"images": image, "bounding_boxes": bounding_boxes}


def load_pascal_voc(split, dataset, bounding_box_format):
    ds = tfds.load(dataset, split=split, with_info=False, shuffle_files=True)
    ds = ds.map(
        lambda x: unpackage_raw_tfds_inputs(x, bounding_box_format=bounding_box_format),
        num_parallel_calls=tf_data.AUTOTUNE,
    )
    return ds


def create_augmenter_fn(augmenters):
    def augmenter_fn(inputs):
        for augmenter in augmenters:
            inputs = augmenter(inputs)
        return inputs
    return augmenter_fn


def apply_augmentation(ds, bounding_box_format="xywh"):
    augmenters = [
        keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format=bounding_box_format),
        keras_cv.layers.JitteredResize(
            target_size=(640, 640), scale_factor=(0.75, 1.3), bounding_box_format=bounding_box_format
        ),
    ]

    augmenter_fn = create_augmenter_fn(augmenters)
    return ds.map(augmenter_fn, num_parallel_calls=tf_data.AUTOTUNE)


def apply_resizing(ds, bounding_box_format="xywh"):
    resizing = keras_cv.layers.Resizing(
        640, 640, bounding_box_format=bounding_box_format, pad_to_aspect_ratio=True
    )
    return ds.map(resizing, num_parallel_calls=tf_data.AUTOTUNE)


def dict_to_tuple(inputs):
    return inputs["images"], bounding_box.to_dense(
        inputs["bounding_boxes"], max_boxes=32
    )


def prepare_dataset(split, dataset, bounding_box_format, batch_size):
    ds = load_pascal_voc(split, dataset, bounding_box_format)
    ds = ds.shuffle(batch_size * 4)
    ds = ds.ragged_batch(batch_size, drop_remainder=True)
    if split == "train":
        ds = apply_augmentation(ds, bounding_box_format)
    else:
        ds = apply_resizing(ds, bounding_box_format)

    ds = ds.map(dict_to_tuple, num_parallel_calls=tf_data.AUTOTUNE)
    ds = ds.prefetch(tf_data.AUTOTUNE)
    return ds
