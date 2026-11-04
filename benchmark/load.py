import os
import shutil
import tempfile
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def dice_coef(y_true, y_pred):
    smooth = 1e-6

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true * y_pred)

    dice = (
        (2.0 * intersection + smooth)
        / (
            tf.reduce_sum(y_true)
            + tf.reduce_sum(y_pred)
            + smooth
        )
    )

    return dice


# Custom binary Dice metric
def binary_dice_coef(y_true, y_pred):
    y_pred = tf.cast(y_pred >= 0.5, tf.float32)
    return dice_coef(y_true, y_pred)


# Custom Dice loss
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


# Combined BCE + Dice loss
def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

    return tf.reduce_mean(bce) + dice_loss(y_true, y_pred)


# Load the saved CNN model and handle legacy HDF5 files named with a .keras suffix.
model_path = os.path.join(
    os.path.dirname(os.path.abspath("cnn_semantic.ipynb")),
    "unet_brain_model_best.keras"
)

if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

custom_objects = {
    "dice_coef": dice_coef,
    "binary_dice_coef": binary_dice_coef,
    "dice_loss": dice_loss,
    "combined_loss": combined_loss,
}

if h5py.is_hdf5(model_path):
    class CompatibleConv2DTranspose(tf.keras.layers.Conv2DTranspose):
        @classmethod
        def from_config(cls, config):
            config = dict(config)
            groups = config.pop("groups", 1)
            if groups != 1:
                raise ValueError(
                    f"Cannot load Conv2DTranspose with groups={groups}"
                )
            return super().from_config(config)

    with tempfile.TemporaryDirectory() as temp_dir:
        legacy_model_path = os.path.join(
            temp_dir,
            "unet_brain_model_best.h5",
        )
        shutil.copy2(model_path, legacy_model_path)
        brain_model = tf.keras.models.load_model(
            legacy_model_path,
            custom_objects={
                **custom_objects,
                "Conv2DTranspose": CompatibleConv2DTranspose,
            },
            compile=False,
        )
else:
    brain_model = tf.keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False,
    )

print("Model input shape:", brain_model.input_shape)
print("Test data shape:", X_test.shape)

y_pred = brain_model.predict(X_test, verbose=0)


# Randomly select distinct test images with non-empty ground-truth masks.
valid_indices = np.flatnonzero(
    np.any(Y_test > 0, axis=(1, 2, 3))
)

if len(valid_indices) == 0:
    raise ValueError("No test images contain a ground-truth mask.")

number_of_examples = min(100, len(valid_indices))

selected_indices = np.random.choice(
    valid_indices,
    size=number_of_examples,
    replace=False,
)

print("Random test images with ground truth:", selected_indices)


plt.style.use("ggplot")
plt.rcParams["figure.facecolor"] = "#171717"
plt.rcParams["text.color"] = "#AAAAAA"


def display_mask_channels_as_rgb(
    mask,
    title="Mask Channels as RGB",
):
    channel_names = [
        "Necrotic (NEC)",
        "Edema (ED)",
        "Tumour (ET)",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(9.75, 5))

    for idx, ax in enumerate(axes):
        rgb_mask = np.zeros(
            (mask.shape[1], mask.shape[2], 3),
            dtype=np.uint8,
        )

        rgb_mask[..., idx] = (
            mask[idx, :, :] * 255
        ).astype(np.uint8)

        ax.imshow(rgb_mask)
        ax.axis("off")
        ax.set_title(channel_names[idx])

    plt.suptitle(title, fontsize=20, y=0.93)
    plt.tight_layout()
    plt.show()


dice_scores = []

for example_number, idx in enumerate(selected_indices, start=1):
    ground_truth = Y_test[idx]
    prediction = y_pred[idx]

    intersection = np.sum(ground_truth * prediction)

    dice_score = (
        (2.0 * intersection + 1e-6)
        / (
            np.sum(ground_truth)
            + np.sum(prediction)
            + 1e-6
        )
    )

    dice_scores.append(dice_score)

    print(
        f"Example {example_number}: image index {idx}, "
        f"Dice score {dice_score:.4f}"
    )

    display_mask_channels_as_rgb(
        ground_truth.transpose(2, 0, 1),
        title=f"Actual Mask - Example {example_number}",
    )

    display_mask_channels_as_rgb(
        prediction.transpose(2, 0, 1),
        title=(
            f"Predicted Model Mask - Example {example_number} "
            f"(Dice: {dice_score:.4f})"
        ),
    )


print(f"\nAverage Dice score: {np.mean(dice_scores):.4f}")

