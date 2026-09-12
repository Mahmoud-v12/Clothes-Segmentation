"""
Preprocessing, augmentation, tf.data.Dataset construction, and
class-weight estimation for the segmentation pipeline.
"""

import numpy as np
import tensorflow as tf

import config


def preprocess_sample(image, mask):
    """Resizes + normalizes a single (PIL image, PIL mask) pair."""
    image = np.array(image.convert("RGB"))
    image = image.astype(np.float32) / 255.0

    mask = np.array(mask)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = mask.astype(np.int32)

    image = tf.image.resize(image, config.IMAGE_SIZE, method="bilinear")

    mask = tf.image.resize(mask[..., None], config.IMAGE_SIZE, method="nearest")
    mask = tf.squeeze(mask, axis=-1)
    mask = tf.cast(mask, tf.int32)

    return image, mask


def augment(image, mask):
    """Random horizontal flip + brightness/contrast jitter (training split only)."""
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask[..., None])
        mask = tf.squeeze(mask, axis=-1)

    image = tf.image.random_brightness(image, max_delta=0.10)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, mask


def data_generator(hf_dataset, training=False):
    """Yields (image, mask) pairs, preprocessed and optionally augmented."""
    for i in range(len(hf_dataset)):
        image = hf_dataset[i]["image"]
        mask = hf_dataset[i]["mask"]

        image, mask = preprocess_sample(image, mask)

        if training:
            image, mask = augment(image, mask)

        yield image, mask


def build_tf_dataset(hf_dataset, training=False, shuffle_buffer=512):
    """Wraps a HuggingFace split into a batched, prefetched tf.data.Dataset."""
    output_signature = (
        tf.TensorSpec(shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH), dtype=tf.int32),
    )

    ds = tf.data.Dataset.from_generator(
        lambda: data_generator(hf_dataset, training=training),
        output_signature=output_signature,
    )

    if training:
        ds = ds.shuffle(shuffle_buffer, seed=config.SEED)

    ds = ds.batch(config.BATCH_SIZE).prefetch(config.AUTOTUNE)
    return ds


def compute_class_weights(hf_dataset, num_classes=None, max_samples=None, seed=None):
    """
    Estimates per-class pixel frequency from a random sample of the
    training set and returns inverse-sqrt-frequency weights (normalized
    to an average of ~1). Used to upweight rare classes in the loss so
    the model isn't dominated by the most common classes.
    """
    num_classes = config.NUM_CLASSES if num_classes is None else num_classes
    max_samples = config.CLASS_WEIGHT_MAX_SAMPLES if max_samples is None else max_samples
    seed = config.SEED if seed is None else seed

    counts = np.zeros(num_classes, dtype=np.float64)

    n = min(max_samples, len(hf_dataset))
    rng = np.random.RandomState(seed)
    idxs = rng.choice(len(hf_dataset), size=n, replace=False)

    for i in idxs:
        mask = np.array(hf_dataset[int(i)]["mask"])
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        vals, cnts = np.unique(mask, return_counts=True)
        for v, c in zip(vals, cnts):
            if 0 <= v < num_classes:
                counts[v] += c

    # Avoid division by zero for classes that never appeared in the sample
    counts = np.clip(counts, 1, None)
    freq = counts / counts.sum()

    # Inverse sqrt frequency: smoother than plain 1/freq, avoids exploding
    # weights for extremely rare classes while still boosting them.
    weights = 1.0 / np.sqrt(freq)
    weights = weights / weights.mean()

    return weights.astype(np.float32)
