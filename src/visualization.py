from keras_cv import visualization
import matplotlib.pyplot as plt


def visualize_detections(model, dataset, bounding_box_format):
    class_ids = [
        "Aeroplane",
        "Bicycle",
        "Bird",
        "Boat",
        "Bottle",
        "Bus",
        "Car",
        "Cat",
        "Chair",
        "Cow",
        "Dining Table",
        "Dog",
        "Horse",
        "Motorbike",
        "Person",
        "Potted Plant",
        "Sheep",
        "Sofa",
        "Train",
        "Tvmonitor",
        "Total",
    ]
    class_mapping = dict(zip(range(len(class_ids)), class_ids))

    visualization_ds = dataset.unbatch()
    visualization_ds = visualization_ds.ragged_batch(16)
    visualization_ds = visualization_ds.shuffle(8)

    images, y_true = next(iter(visualization_ds.take(1)))
    y_pred = model.predict(images)
    visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        bounding_box_format=bounding_box_format,
        y_true=y_true,
        y_pred=y_pred,
        scale=4,
        rows=2,
        cols=2,
        show=True,
        font_scale=0.7,
        class_mapping=class_mapping,
    )
    plt.show()