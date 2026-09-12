"""
Loss functions and the mean-IoU metric used to train/evaluate the
segmentation model: a class-weighted combination of focal loss and
Dice loss, plus a mean-IoU metric usable inside model.fit / evaluate.
"""

import tensorflow as tf

import config


def multiclass_dice_loss(y_true, y_pred, smooth=1e-6, class_weights=None):
    y_true = tf.cast(y_true, tf.int32)
    y_true_one_hot = tf.one_hot(y_true, depth=config.NUM_CLASSES)
    y_true_one_hot = tf.cast(y_true_one_hot, tf.float32)

    y_true_flat = tf.reshape(y_true_one_hot, [tf.shape(y_true_one_hot)[0], -1, config.NUM_CLASSES])
    y_pred_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1, config.NUM_CLASSES])

    intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=1)
    denominator = tf.reduce_sum(y_true_flat, axis=1) + tf.reduce_sum(y_pred_flat, axis=1)

    dice_per_class = (2.0 * intersection + smooth) / (denominator + smooth)

    if class_weights is not None:
        # Weighted average across classes instead of a plain mean, so
        # rare classes contribute more to the loss.
        weighted_dice = tf.reduce_sum(dice_per_class * class_weights, axis=-1) / tf.reduce_sum(class_weights)
    else:
        weighted_dice = tf.reduce_mean(dice_per_class, axis=-1)

    return 1.0 - tf.reduce_mean(weighted_dice)


def weighted_focal_loss(y_true, y_pred, gamma=2.0, class_weights=None):
    # Focal loss down-weights easy, already-correct pixels and focuses
    # training on hard / misclassified ones -- important on an
    # imbalanced 18-class mask where a few dominant classes would
    # otherwise dominate the gradient.
    y_true = tf.cast(y_true, tf.int32)
    y_true_one_hot = tf.one_hot(y_true, depth=config.NUM_CLASSES)
    y_true_one_hot = tf.cast(y_true_one_hot, tf.float32)

    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
    modulating_factor = tf.pow(1.0 - y_pred, gamma)
    focal = modulating_factor * cross_entropy

    if class_weights is not None:
        focal = focal * class_weights

    focal = tf.reduce_sum(focal, axis=-1)
    return tf.reduce_mean(focal)


def make_combined_loss(class_weights_tf):
    """
    Returns a combined_loss(y_true, y_pred) closure bound to the given
    per-class weights, ready to pass to model.compile(loss=...).

    Dice correlates directly with IoU, so it's weighted equally with
    focal loss (0.5 / 0.5) to push mIoU up specifically.
    """
    def combined_loss(y_true, y_pred):
        focal = weighted_focal_loss(y_true, y_pred, gamma=2.0, class_weights=class_weights_tf)
        dice = multiclass_dice_loss(y_true, y_pred, class_weights=class_weights_tf)
        return 0.5 * focal + 0.5 * dice

    return combined_loss


def mean_iou_metric(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, tf.int32)
    y_true = tf.cast(y_true, tf.int32)

    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    confusion_matrix = tf.math.confusion_matrix(
        y_true, y_pred, num_classes=config.NUM_CLASSES, dtype=tf.float32
    )

    intersection = tf.linalg.diag_part(confusion_matrix)
    ground_truth_pixels = tf.reduce_sum(confusion_matrix, axis=1)
    predicted_pixels = tf.reduce_sum(confusion_matrix, axis=0)
    union = ground_truth_pixels + predicted_pixels - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)

    # Ignore classes that don't exist in the ground truth for this batch
    valid = union > 0
    iou = tf.boolean_mask(iou, valid)

    return tf.reduce_mean(iou)
