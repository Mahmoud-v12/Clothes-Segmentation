import random

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

import config
import data_prep
import data_processing
import losses


def load_trained_model(model_path=None):
    """
    Loads a saved model. class_weights only affect the loss during
    training, so a dummy (all-ones) weighting is used here -- it does
    not change evaluation or inference results.
    """
    model_path = config.FINAL_MODEL_PATH if model_path is None else model_path

    dummy_weights = tf.ones(config.NUM_CLASSES, dtype=tf.float32)
    combined_loss = losses.make_combined_loss(dummy_weights)

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "combined_loss": combined_loss,
            "mean_iou_metric": losses.mean_iou_metric,
        },
    )
    print("Model loaded successfully.")
    return model


def calculate_per_class_iou(y_true, y_pred, num_classes=None):
    num_classes = config.NUM_CLASSES if num_classes is None else num_classes
    ious = []
    for class_id in range(num_classes):
        true_class = y_true == class_id
        pred_class = y_pred == class_id

        intersection = np.logical_and(true_class, pred_class).sum()
        union = np.logical_or(true_class, pred_class).sum()

        ious.append(np.nan if union == 0 else intersection / union)

    return np.array(ious)


def evaluate_on_test_set(model, test_tf):
    """Runs model.evaluate and returns the metrics dict (loss, pixel accuracy, mIoU)."""
    results = model.evaluate(test_tf, return_dict=True)
    print(results)
    return results


def per_class_iou_report(model, test_tf):
    """Collects predictions over the whole test set and prints per-class IoU."""
    all_true, all_pred = [], []

    for images_batch, masks_batch in test_tf:
        probs = model.predict(images_batch, verbose=0)
        preds = np.argmax(probs, axis=-1)

        all_true.append(masks_batch.numpy())
        all_pred.append(preds)

    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)

    class_ious = calculate_per_class_iou(all_true, all_pred)

    for class_id, iou in enumerate(class_ious):
        label = "N/A" if np.isnan(iou) else f"{iou:.4f}"
        print(f"Class {class_id}: IoU = {label}")

    mean_iou_score = np.nanmean(class_ious)
    print("\nMean IoU:", mean_iou_score)

    return class_ious, mean_iou_score


def predict_image(model, image_path):
    """Runs inference on a single external image file. Returns (original, pred_mask)."""
    image = Image.open(image_path).convert("RGB")
    original = np.array(image)

    image_resized = tf.image.resize(original, config.IMAGE_SIZE, method="bilinear")
    image_resized = tf.cast(image_resized, tf.float32) / 255.0
    image_batch = tf.expand_dims(image_resized, axis=0)

    prediction = model.predict(image_batch, verbose=0)[0]
    pred_mask = np.argmax(prediction, axis=-1)

    return original, pred_mask


def plot_qualitative_samples(model, test_data, num_samples=3, seed=None):
    """Original vs Ground Truth vs Prediction for a few random test-set samples."""
    seed = config.SEED if seed is None else seed
    rng = random.Random(seed)
    sample_indices = rng.sample(range(len(test_data)), num_samples)

    plt.figure(figsize=(12, 4 * num_samples))

    for row, idx in enumerate(sample_indices):
        image, mask = data_processing.preprocess_sample(
            test_data[idx]["image"], test_data[idx]["mask"]
        )
        image_batch = tf.expand_dims(image, axis=0)

        pred = model.predict(image_batch, verbose=0)[0]
        pred_mask = np.argmax(pred, axis=-1)

        plt.subplot(num_samples, 3, row * 3 + 1)
        plt.imshow(image.numpy())
        plt.title(f"Original (test idx {idx})")
        plt.axis("off")

        plt.subplot(num_samples, 3, row * 3 + 2)
        plt.imshow(mask.numpy(), cmap="tab20", vmin=0, vmax=config.NUM_CLASSES - 1)
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(num_samples, 3, row * 3 + 3)
        plt.imshow(pred_mask, cmap="tab20", vmin=0, vmax=config.NUM_CLASSES - 1)
        plt.title("Prediction")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    _, _, test_data = data_prep.load_and_split_dataset()
    test_tf = data_processing.build_tf_dataset(test_data, training=False)

    model = load_trained_model()

    evaluate_on_test_set(model, test_tf)
    per_class_iou_report(model, test_tf)
    plot_qualitative_samples(model, test_data)


if __name__ == "__main__":
    main()
