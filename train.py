"""
Trains the DeepLabV3+ (ResNet50) clothes/body segmentation model.

Usage:
    python train.py

All hyperparameters and paths live in config.py.
"""

import random

import numpy as np
import tensorflow as tf

import config
import data_prep
import data_processing
import losses
import model as model_module


def set_seeds(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main():
    set_seeds()

    print("TensorFlow:", tf.__version__)
    print("Num GPUs:", len(tf.config.list_physical_devices("GPU")))

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_data, val_data, test_data = data_prep.load_and_split_dataset()
    print("Train:", len(train_data))
    print("Validation:", len(val_data))
    print("Test:", len(test_data))

    train_tf = data_processing.build_tf_dataset(train_data, training=True)
    val_tf = data_processing.build_tf_dataset(val_data, training=False)

    # ------------------------------------------------------------------
    # Class weights + loss (weights estimated from a sample of train_data)
    # ------------------------------------------------------------------
    class_weights = data_processing.compute_class_weights(train_data)
    class_weights_tf = tf.constant(class_weights, dtype=tf.float32)

    print("Per-class weights:")
    for class_id, w in enumerate(class_weights):
        print(f"  Class {class_id}: {w:.3f}")

    combined_loss = losses.make_combined_loss(class_weights_tf)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = model_module.build_deeplabv3_resnet50()
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss=combined_loss,
        metrics=["sparse_categorical_accuracy", losses.mean_iou_metric],
    )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            config.BEST_MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.MIN_LR,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    history = model.fit(
        train_tf,
        validation_data=val_tf,
        epochs=config.EPOCHS,
        callbacks=callbacks,
    )

    model.save(config.FINAL_MODEL_PATH)
    print("Model saved successfully to", config.FINAL_MODEL_PATH)

    return history


if __name__ == "__main__":
    main()
