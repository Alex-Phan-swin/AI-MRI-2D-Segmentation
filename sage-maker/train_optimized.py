
import os
import json
from itertools import product

import numpy as np
import h5py
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from split_utils import split_files_by_patient


# ============================================================
# Configuration
# ============================================================

SEED = 14

# EPOCHS = 250

# BATCH_SIZE = 32

# LEARNING_RATE = 0.0001


# ============================================================
# Hyperparameter tuning
# ============================================================
# These combinations are evaluated automatically in sequence.
HYPERPARAMETER_TUNING = {
    "EPOCHS": [150],
    "BATCH_SIZE": [16, 32, 64],
    "LEARNING_RATE": [1e-4, 5e-5, 1e-5],
}

HYPERPARAMETER_GRID = [
    dict(zip(HYPERPARAMETER_TUNING.keys(), values))
    for values in product(
        *HYPERPARAMETER_TUNING.values()
    )
]

ACTIVE_HYPERPARAMETERS = HYPERPARAMETER_GRID[0]
EPOCHS = ACTIVE_HYPERPARAMETERS["EPOCHS"]
BATCH_SIZE = ACTIVE_HYPERPARAMETERS["BATCH_SIZE"]
LEARNING_RATE = ACTIVE_HYPERPARAMETERS["LEARNING_RATE"]


# ============================================================
# Reproducibility
# ============================================================

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# SageMaker directories
# ============================================================

DATA_DIRECTORY = os.environ.get(
    "SM_CHANNEL_TRAIN",
    "./data"
)

MODEL_DIRECTORY = os.environ.get(
    "SM_MODEL_DIR",
    "./model"
)

os.makedirs(MODEL_DIRECTORY, exist_ok=True)


# ============================================================
# TensorFlow device information
# ============================================================

print("=" * 60)
print("Training configuration")
print("=" * 60)

print("Data directory:", DATA_DIRECTORY)
print("Model directory:", MODEL_DIRECTORY)

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("GPU available:", gpus)

    for gpu in gpus:
        print("GPU:", gpu)
else:
    print("GPU available: No")
    print("Using CPU")

print("=" * 60)


# ============================================================
# Find H5 files
# ============================================================

def get_h5_files(directory):

    files = []

    for file_name in os.listdir(directory):

        if file_name.endswith(".h5"):

            files.append(
                os.path.join(
                    directory,
                    file_name
                )
            )

    files.sort()

    return files


all_files = get_h5_files(
    DATA_DIRECTORY
)

print(
    "Total H5 files found:",
    len(all_files)
)


if len(all_files) == 0:

    raise RuntimeError(
        f"No .h5 files found in {DATA_DIRECTORY}"
    )


# ============================================================
# Patient-level train / validation / test split
# ============================================================

train_files, validation_files, test_files, split_manifest = split_files_by_patient(
    all_files,
    seed=SEED,
)

split_path = os.path.join(
    MODEL_DIRECTORY,
    "patient_split.json"
)
with open(split_path, "w", encoding="utf-8") as file:
    json.dump(split_manifest, file, indent=2)


print("=" * 60)
print("Dataset split")
print("=" * 60)

print(
    "Total files:",
    len(all_files)
)

print(
    "Training patients / files:",
    split_manifest["patient_counts"]["train"],
    "/",
    len(train_files)
)

print(
    "Validation patients / files:",
    split_manifest["patient_counts"]["validation"],
    "/",
    len(validation_files)
)

print(
    "Untouched test patients / files:",
    split_manifest["patient_counts"]["test"],
    "/",
    len(test_files)
)

print("Patient leakage check: PASS")
print("Saved exact patient split:", split_path)
print("=" * 60)


# ============================================================
# Determine image and mask dimensions
# ============================================================

with h5py.File(
    train_files[0],
    "r"
) as h5_file:

    first_image = np.asarray(
        h5_file["image"],
        dtype=np.float32
    )

    first_mask = np.asarray(
        h5_file["mask"],
        dtype=np.float32
    )


input_height = first_image.shape[0]

input_width = first_image.shape[1]

input_channels = first_image.shape[-1]

output_channels = first_mask.shape[-1]


print("Image height:", input_height)
print("Image width:", input_width)
print("Input channels:", input_channels)
print("Output channels:", output_channels)


# ============================================================
# TensorFlow Dataset Generator
# ============================================================

class BraTSDataset(
    keras.utils.Sequence
):

    def __init__(
        self,
        files,
        batch_size,
        shuffle=False
    ):

        self.files = files

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.indexes = np.arange(
            len(self.files)
        )

        self.on_epoch_end()


    def __len__(self):

        return int(
            np.ceil(
                len(self.files)
                / self.batch_size
            )
        )


    def __getitem__(
        self,
        index
    ):

        start = (
            index
            * self.batch_size
        )

        end = min(
            start
            + self.batch_size,
            len(self.files)
        )

        batch_indexes = self.indexes[
            start:end
        ]

        batch_files = [
            self.files[i]
            for i in batch_indexes
        ]


        images = []

        masks = []


        for file_path in batch_files:

            with h5py.File(
                file_path,
                "r"
            ) as h5_file:

                image = np.asarray(
                    h5_file["image"],
                    dtype=np.float32
                )

                mask = np.asarray(
                    h5_file["mask"],
                    dtype=np.float32
                )


            # image: H x W x 4
            # mask:  H x W x 3

            images.append(image)

            masks.append(mask)


        images = np.asarray(
            images,
            dtype=np.float32
        )

        masks = np.asarray(
            masks,
            dtype=np.float32
        )


        return images, masks


    def on_epoch_end(self):

        if self.shuffle:

            np.random.shuffle(
                self.indexes
            )


# ============================================================
# Create datasets
# ============================================================
# Datasets are created per hyperparameter run below.

# ============================================================
# U-Net model
# ============================================================

def convolution_block(
    x,
    filters
):

    x = layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        activation="relu"
    )(x)

    x = layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        activation="relu"
    )(x)

    return x


def create_unet(
    input_channels,
    output_channels
):

    inputs = keras.Input(
        shape=(
            input_height,
            input_width,
            input_channels
        ),
        name="mri_input"
    )


    # ========================================================
    # Encoder - Level 1
    # ========================================================

    encoder1 = convolution_block(
        inputs,
        32
    )

    pool1 = layers.MaxPooling2D(
        pool_size=2
    )(encoder1)


    # ========================================================
    # Encoder - Level 2
    # ========================================================

    encoder2 = convolution_block(
        pool1,
        64
    )

    pool2 = layers.MaxPooling2D(
        pool_size=2
    )(encoder2)


    # ========================================================
    # Bottleneck
    # ========================================================

    bottleneck = convolution_block(
        pool2,
        128
    )


    # ========================================================
    # Decoder - Level 2
    # ========================================================

    up2 = layers.Conv2DTranspose(
        64,
        kernel_size=2,
        strides=2,
        padding="same"
    )(bottleneck)

    concat2 = layers.Concatenate()([
        up2,
        encoder2
    ])

    decoder2 = convolution_block(
        concat2,
        64
    )


    # ========================================================
    # Decoder - Level 1
    # ========================================================

    up1 = layers.Conv2DTranspose(
        32,
        kernel_size=2,
        strides=2,
        padding="same"
    )(decoder2)

    concat1 = layers.Concatenate()([
        up1,
        encoder1
    ])

    decoder1 = convolution_block(
        concat1,
        32
    )


    # ========================================================
    # Segmentation output
    # ========================================================

    outputs = layers.Conv2D(
        output_channels,
        kernel_size=1,
        padding="same",
        activation="sigmoid",
        name="segmentation_output"
    )(decoder1)


    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="BrainMRI_UNet"
    )


    return model


# ============================================================
# Dice coefficient metric
# ============================================================
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


# ============================================================
# Hyperparameter run wrapper
# ============================================================

def run_training_run(run_name, epochs, batch_size, learning_rate):

    print("=" * 60)
    print(f"Starting hyperparameter run: {run_name}")
    print(f"EPOCHS={epochs}, BATCH_SIZE={batch_size}, LEARNING_RATE={learning_rate}")
    print("=" * 60)

    train_dataset = BraTSDataset(
        train_files,
        batch_size=batch_size,
        shuffle=True
    )

    validation_dataset = BraTSDataset(
        validation_files,
        batch_size=batch_size,
        shuffle=False
    )

    brain_model = create_unet(
        input_channels=input_channels,
        output_channels=output_channels
    )

    print("=" * 60)
    print("U-Net model")
    print("=" * 60)
    brain_model.summary()
    print("=" * 60)

    brain_model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            dice_coef,
        ],
    )

    run_model_dir = os.path.join(MODEL_DIRECTORY, run_name)
    os.makedirs(run_model_dir, exist_ok=True)

    best_model_path = os.path.join(
        run_model_dir,
        "unet_brain_model_best.keras"
    )

    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=best_model_path,
        monitor="val_dice_coef",
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor="val_dice_coef",
        mode="max",
        patience=20,
        restore_best_weights=True,
        verbose=1,
    )

    history_path = os.path.join(
        run_model_dir,
        "unet_training_history.json"
    )

    print("=" * 60)
    print("Starting U-Net training")
    print("=" * 60)

    history = brain_model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
        ],
        verbose=1,
    )

    final_model_path = os.path.join(
        run_model_dir,
        "unet_brain_model.keras"
    )

    brain_model.save(final_model_path)

    with open(history_path, "w") as file:
        json.dump(history.history, file)

    best_model = keras.models.load_model(
        best_model_path,
        custom_objects={"dice_coef": dice_coef},
    )

    best_epoch_index = int(np.argmax(history.history["val_dice_coef"]))
    best_epoch = best_epoch_index + 1
    validation_accuracy = history.history.get(
        "val_binary_accuracy",
        history.history.get("val_accuracy", [None]),
    )

    if validation_accuracy:
        validation_accuracy = float(validation_accuracy[best_epoch_index])
    else:
        validation_accuracy = None

    validation_dice = float(history.history["val_dice_coef"][best_epoch_index])
    print(f"Best validation dice: {validation_dice:.4f}")
    print(
        f"Best-epoch validation accuracy: {validation_accuracy:.4f}"
        if validation_accuracy is not None
        else "Validation accuracy: N/A"
    )

    validation_results_path = os.path.join(
        run_model_dir,
        "validation_results.json"
    )

    validation_results = {
        "validation_dice": validation_dice,
        "validation_accuracy": validation_accuracy,
        "best_epoch": best_epoch,
        "patient_split": "patient_split.json",
        "patient_counts": split_manifest["patient_counts"],
        "sample_counts": split_manifest["sample_counts"],
        "patient_leakage": split_manifest["patient_leakage"],
    }

    with open(validation_results_path, "w") as file:
        json.dump(validation_results, file, indent=4)

    print("=" * 60)
    print("U-Net training complete.")
    print("Best model:", best_model_path)
    print("Final model:", final_model_path)
    print("Training history:", history_path)
    print("Validation results:", validation_results_path)
    print("=" * 60)

    return {
        **validation_results,
        "run_name": run_name,
        "best_model_path": best_model_path,
        "batch_size": batch_size,
    }


# ============================================================
# Automated hyperparameter tuning
# ============================================================

print("=" * 60)
print("Hyperparameter search grid")
print("=" * 60)
print("Total configurations:", len(HYPERPARAMETER_GRID))
for config in HYPERPARAMETER_GRID:
    print(config)
print("=" * 60)

run_results = []
for run_index, config in enumerate(HYPERPARAMETER_GRID, start=2):
    run_name = (
        f"run_{run_index}_"
        f"epochs_{config['EPOCHS']}_"
        f"batch_{config['BATCH_SIZE']}_"
        f"lr_{config['LEARNING_RATE']}"
    )
    run_results.append(
        run_training_run(
            run_name=run_name,
            epochs=config["EPOCHS"],
            batch_size=config["BATCH_SIZE"],
            learning_rate=config["LEARNING_RATE"],
        )
    )

best_run = max(run_results, key=lambda result: result["validation_dice"])
print("=" * 60)
print("Hyperparameter selection complete")
print("Selected run:", best_run["run_name"])
print(f"Best validation dice: {best_run['validation_dice']:.4f}")
print("=" * 60)

test_dataset = BraTSDataset(
    test_files,
    batch_size=best_run["batch_size"],
    shuffle=False,
)
best_model = keras.models.load_model(
    best_run["best_model_path"],
    custom_objects={"dice_coef": dice_coef},
)
test_results = best_model.evaluate(
    test_dataset,
    verbose=1,
    return_dict=True,
)
test_summary = {
    "selected_run": best_run["run_name"],
    "selection_metric": "validation_dice",
    "test_metrics": {
        name: float(value)
        for name, value in test_results.items()
    },
    "patient_split": "patient_split.json",
    "patient_counts": split_manifest["patient_counts"],
    "sample_counts": split_manifest["sample_counts"],
    "patient_leakage": split_manifest["patient_leakage"],
}
test_results_path = os.path.join(MODEL_DIRECTORY, "selected_model_test_results.json")
with open(test_results_path, "w") as file:
    json.dump(test_summary, file, indent=4)
print("Evaluated test set once for selected model:", test_results_path)
