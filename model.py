from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50

import config


def aspp_block(x, filters=256):
    """Atrous Spatial Pyramid Pooling: dilated rates 6/12/18 + global pooling branch."""
    input_shape = x.shape
    h, w = input_shape[1], input_shape[2]

    b0 = layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
    b0 = layers.BatchNormalization()(b0)
    b0 = layers.ReLU()(b0)

    b1 = layers.Conv2D(filters, 3, padding="same", dilation_rate=6, use_bias=False)(x)
    b1 = layers.BatchNormalization()(b1)
    b1 = layers.ReLU()(b1)

    b2 = layers.Conv2D(filters, 3, padding="same", dilation_rate=12, use_bias=False)(x)
    b2 = layers.BatchNormalization()(b2)
    b2 = layers.ReLU()(b2)

    b3 = layers.Conv2D(filters, 3, padding="same", dilation_rate=18, use_bias=False)(x)
    b3 = layers.BatchNormalization()(b3)
    b3 = layers.ReLU()(b3)

    b4 = layers.GlobalAveragePooling2D()(x)
    b4 = layers.Reshape((1, 1, input_shape[-1]))(b4)
    b4 = layers.Conv2D(filters, 1, padding="same", use_bias=False)(b4)
    b4 = layers.BatchNormalization()(b4)
    b4 = layers.ReLU()(b4)
    b4 = layers.Resizing(h, w, interpolation="bilinear")(b4)

    x = layers.Concatenate()([b0, b1, b2, b3, b4])
    x = layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.1)(x)

    return x


def build_deeplabv3_resnet50(input_shape=None, num_classes=None):
    """Builds the DeepLabV3+ segmentation model with an ImageNet-pretrained ResNet50 backbone."""
    input_shape = input_shape or (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3)
    num_classes = num_classes or config.NUM_CLASSES

    inputs = layers.Input(shape=input_shape)

    backbone = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)

    # High-level features for ASPP (stride 16)
    high_level = backbone.get_layer("conv4_block6_out").output

    # Low-level features for the skip connection (stride 4) -- this is
    # the key addition that turns DeepLabV3 into DeepLabV3+
    low_level = backbone.get_layer("conv2_block3_out").output

    x = aspp_block(high_level, filters=256)

    # Upsample ASPP output up to the low-level feature map's resolution
    low_h, low_w = low_level.shape[1], low_level.shape[2]
    x = layers.Resizing(low_h, low_w, interpolation="bilinear")(x)

    # Project low-level features to fewer channels before concatenating
    # (48 channels, as in the original DeepLabV3+ paper -- keeps the skip
    # connection from dominating the richer ASPP features)
    low_level_proj = layers.Conv2D(48, 1, padding="same", use_bias=False)(low_level)
    low_level_proj = layers.BatchNormalization()(low_level_proj)
    low_level_proj = layers.ReLU()(low_level_proj)

    # Fuse high-level context with low-level spatial detail
    x = layers.Concatenate()([x, low_level_proj])

    # Refine the fused features
    x = layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Upsample back to original image size
    x = layers.Resizing(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, interpolation="bilinear")(x)

    outputs = layers.Conv2D(num_classes, 1, padding="same", activation="softmax")(x)

    return Model(inputs=inputs, outputs=outputs, name="DeepLabV3Plus_ResNet50")
