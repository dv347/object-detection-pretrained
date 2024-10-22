import keras
import keras_cv


def create_model(base_lr=0.005, bounding_box_format="xywh"):
    optimizer = keras.optimizers.SGD(
        learning_rate=base_lr, momentum=0.9, global_clipnorm=10.0
    )

    model = keras_cv.models.YOLOV8Detector.from_preset(
        "resnet50_imagenet",
        bounding_box_format=bounding_box_format,
        num_classes=20,
    )

    model.compile(
        classification_loss="binary_crossentropy",
        box_loss="ciou",
        optimizer=optimizer,
    )

    return model


def get_callbacks(eval_ds, save_path='model_checkpoints/yolov8_best_weights.weights.h5', bounding_box_format="xywh"):
    coco_metrics_callback = keras_cv.callbacks.PyCOCOCallback(
        eval_ds.take(20), bounding_box_format=bounding_box_format
    )
    
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=save_path,
        save_weights_only=True,
        monitor='loss', 
        mode='min', 
        save_best_only=True
    )

    return [coco_metrics_callback, checkpoint_callback]


def prepare_for_inference(model, iou_threshold=0.5, confidence_threshold=0.75, bounding_box_format="xywh"):
    model.prediction_decoder = keras_cv.layers.NonMaxSuppression(
        bounding_box_format=bounding_box_format,
        from_logits=True,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold,
    )
    return model