from config import BASE_LR, BATCH_SIZE, BOUNDING_BOX_FORMAT, EPOCHS
from data_loader import prepare_dataset
from model import create_model, get_callbacks, prepare_for_inference
from visualization import visualize_detections


if __name__ == "__main__":
    # Load and prepare the training dataset with augmentations
    train_ds = prepare_dataset(
        split="train", 
        dataset="voc/2007", 
        bounding_box_format=BOUNDING_BOX_FORMAT, 
        batch_size=BATCH_SIZE
    )

    # Load and prepare the evaluation dataset with resizing for consistency
    eval_ds = prepare_dataset(
        split="test", 
        dataset="voc/2007", 
        bounding_box_format=BOUNDING_BOX_FORMAT, 
        batch_size=BATCH_SIZE
    )

    # Create and compile the model (YOLOv8 with ResNet-50 backbone)
    model = create_model(base_lr=BASE_LR)

    # Set up callbacks for tracking metrics and saving the best model
    callbacks = get_callbacks(eval_ds)

    # Train the model
    model.fit(
        train_ds.take(20),
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Add Non-Max Suppression for bounding box post-processing
    model = prepare_for_inference(model, bounding_box_format=BOUNDING_BOX_FORMAT)

    # Visualize model detections on the evaluation dataset
    visualize_detections(model, dataset=eval_ds, bounding_box_format=BOUNDING_BOX_FORMAT)