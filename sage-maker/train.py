
import os
import json

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

EPOCHS = 400

BATCH_SIZE = 32

LEARNING_RATE = 0.001


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


#Preprocessing, apply intensity clipping and normalisation
def preprocess_mri_image(image):
    image = np.asarray(image, dtype=np.float32)

    if image.ndim != 3:
        raise ValueError(
            f"Expected image shape H x W x C, got {image.shape}"
        )


    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    brain_mask = np.any(image != 0, axis=-1)

    processed_image = np.zeros_like(
        image,
        dtype=np.float32
    )

    # Process each MRI modality independently.
    for modality in range(image.shape[-1]):

        modality_image = image[:, :, modality]


        brain_values = modality_image[brain_mask]


        if brain_values.size == 0:
            continue


        finite_values = brain_values[np.isfinite(brain_values)]

        if finite_values.size == 0:
            continue

        # ----------------------------------------------------
        # 1. Intensity clipping: 0.5th to 99.5th percentile
        # ----------------------------------------------------

        lower_percentile = np.percentile(
            finite_values,
            0.5
        )

        upper_percentile = np.percentile(
            finite_values,
            99.5
        )

        # Clip this modality independently.
        clipped = np.clip(
            modality_image,
            lower_percentile,
            upper_percentile
        )

        # ----------------------------------------------------
        # 2. Intensity normalisation: z-score
        # ----------------------------------------------------
        clipped_brain_values = clipped[brain_mask]
        finite_clipped_values = clipped_brain_values[np.isfinite(clipped_brain_values)]

        if finite_clipped_values.size == 0:
            continue

        mean = np.mean(finite_clipped_values,dtype=np.float64)
        std = np.std(finite_clipped_values,dtype=np.float64)


        # Avoid division by zero for constant-valued modalities.
        if std < 1e-8:
            normalised = np.zeros_like(modality_image,dtype=np.float32)
        else:
            normalised = (clipped.astype(np.float64) - mean) / std

            normalised = normalised.astype(np.float32)


        normalised[~brain_mask] = 0.0
        processed_image[:, :, modality] = normalised

    return processed_image

#data augmentation applies variance and intensity scaling and mirroring
def augment_mri_image(image, mask):

    image = image.copy()
    mask = mask.copy()

    # Identify background before intensity augmentation.
    brain_mask = np.any(image != 0, axis=-1)

    # --------------------------------------------------------
    # 1. Variance shift: probability 20%
    # --------------------------------------------------------
    if np.random.random() < 0.2:

        variance_factor = 1.0 + np.random.uniform(
            -0.1, 0.1
        )

        # Adjust deviations from the mean, changing variance
        # while approximately preserving the mean.
        for modality in range(image.shape[-1]):

            channel = image[:, :, modality]
            brain_values = channel[brain_mask]

            if brain_values.size == 0:
                continue

            mean = np.mean(brain_values)

            image[:, :, modality] = (
                mean + (channel - mean) * variance_factor
            )

    # --------------------------------------------------------
    # 2. Intensity scaling: probability 20%
    # --------------------------------------------------------
    if np.random.random() < 0.2:

        intensity_factor = np.random.uniform(0.9, 1.1)

        image *= intensity_factor

    # Keep background at zero.
    image[~brain_mask] = 0.0

    # --------------------------------------------------------
    # 3. Mirroring: probability 50% per spatial axis
    # --------------------------------------------------------

    # Axis 0 = height
    if np.random.random() < 0.5:
        image = np.flip(image, axis=0)
        mask = np.flip(mask, axis=0)

    # Axis 1 = width
    if np.random.random() < 0.5:
        image = np.flip(image, axis=1)
        mask = np.flip(mask, axis=1)

    # Return contiguous arrays to avoid negative-stride issues.
    return (
        np.ascontiguousarray(image, dtype=np.float32),
        np.ascontiguousarray(mask, dtype=np.float32)
    )



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

if input_height % 4 != 0 or input_width % 4 != 0:
    raise ValueError(
        "Image height and width must be divisible by 4 for the U-Net "
        f"pooling and upsampling layers; got {input_height} x {input_width}"
    )


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
        shuffle=False,
        augment=False
    ):

        self.files = files

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.augment = augment

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

            #apply preprocessing
            image = preprocess_mri_image(image)

            # 2. Apply augmentation to training data only.
            if self.augment:

                image, mask = augment_mri_image(
                    image, mask
                )

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

train_dataset = BraTSDataset(
    train_files,
    batch_size=BATCH_SIZE,
    shuffle=True,
    augment=True
)


validation_dataset = BraTSDataset(
    validation_files,
    batch_size=BATCH_SIZE,
    shuffle=False,
    augment=False
)


test_dataset = BraTSDataset(
    test_files,
    batch_size=BATCH_SIZE,
    shuffle=False,
    augment=False
)

# ============================================================
# Attention Gate
# ============================================================

def attention_gate(
    skip_connection,
    gating_signal,
    filters
):

    # --------------------------------------------------------
    # Transform skip connection
    # --------------------------------------------------------

    theta_x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same"
    )(skip_connection)

    # --------------------------------------------------------
    # Transform decoder/gating signal
    # --------------------------------------------------------

    phi_g = layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same"
    )(gating_signal)

    # --------------------------------------------------------
    # Combine encoder and decoder information
    # --------------------------------------------------------

    combined = layers.Add()([
        theta_x,
        phi_g
    ])

    combined = layers.Activation(
        "relu"
    )(combined)

    # --------------------------------------------------------
    # Generate attention map
    # --------------------------------------------------------

    attention = layers.Conv2D(
        1,
        kernel_size=1,
        padding="same"
    )(combined)

    attention = layers.Activation(
        "sigmoid"
    )(attention)

    # --------------------------------------------------------
    # Apply attention to skip connection
    # --------------------------------------------------------

    attended_features = layers.Multiply()([
        skip_connection,
        attention
    ])

    return attended_features


# ============================================================
# Convolution block
# ============================================================

def convolution_block(
    x,
    filters
):

    x = layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation(
        "relu"
    )(x)


    x = layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation(
        "relu"
    )(x)

    return x


# ============================================================
# Attention U-Net
# ============================================================

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

    bottleneck = layers.Dropout(
        0.2
    )(bottleneck)


    # ========================================================
    # Decoder - Level 2
    # ========================================================

    up2 = layers.Conv2DTranspose(
        64,
        kernel_size=2,
        strides=2,
        padding="same"
    )(bottleneck)


    # --------------------------------------------------------
    # Attention gate for encoder level 2
    # --------------------------------------------------------

    attention2 = attention_gate(
        encoder2,
        up2,
        64
    )


    concat2 = layers.Concatenate()([
        up2,
        attention2
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


    # --------------------------------------------------------
    # Attention gate for encoder level 1
    # --------------------------------------------------------

    attention1 = attention_gate(
        encoder1,
        up1,
        32
    )


    concat1 = layers.Concatenate()([
        up1,
        attention1
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


    # ========================================================
    # Create model
    # ========================================================

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="BrainMRI_Attention_UNet"
    )


    return model


# ============================================================
# Create Attention U-Net
# ============================================================

brain_model = create_unet(
    input_channels=input_channels,
    output_channels=output_channels
)


print("=" * 60)
print("Attention U-Net model")
print("=" * 60)

brain_model.summary()

print("=" * 60)



# ============================================================
# Dice coefficient metric
# ============================================================  

# def dice_coef(y_true, y_pred):
#     smooth = 1e-6

#     y_true = tf.cast(y_true, tf.float32)
#     y_pred = tf.cast(y_pred, tf.float32)

#     # Calculate Dice separately for each image.
#     # Shape before reduction: (batch, height, width, channels)
#     axes = [1, 2, 3]

#     intersection = tf.reduce_sum(
#         y_true * y_pred, axis=axes
#     )

#     true_sum = tf.reduce_sum(y_true, axis=axes)
#     pred_sum = tf.reduce_sum(y_pred, axis=axes)

#     dice = (2.0 * intersection + smooth) / (
#         true_sum + pred_sum + smooth
#     )

#     # Average the individual image Dice scores.
#     return tf.reduce_mean(dice)


def dice_coef(y_true, y_pred):
    smooth = 1e-6

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Calculate Dice separately for each image.
    axes = [1, 2, 3]

    intersection = tf.reduce_sum(
        y_true * y_pred, axis=axes
    )

    true_sum = tf.reduce_sum(y_true, axis=axes)
    pred_sum = tf.reduce_sum(y_pred, axis=axes)

    dice = (2.0 * intersection + smooth) / (
        true_sum + pred_sum + smooth
    )

    # Identify images with at least one tumour pixel.
    has_tumour = tf.reduce_sum(
        y_true, axis=axes
    ) > 0

    # Keep only images with ground-truth tumours.
    tumour_dice = tf.boolean_mask(
        dice, has_tumour
    )

    # Return 0 if the batch has no tumour-containing images.
    return tf.cond(
        tf.size(tumour_dice) > 0,
        lambda: tf.reduce_mean(tumour_dice),
        lambda: tf.constant(0.0, dtype=tf.float32)
    )


def binary_dice_coef(y_true, y_pred):
    y_pred = tf.cast(y_pred >= 0.5, tf.float32)
    return dice_coef(y_true, y_pred)


# ============================================================
# Dice loss
# ============================================================
def dice_loss(y_true, y_pred):
    smooth = 1e-6

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    axes = [1, 2, 3]

    intersection = tf.reduce_sum(
        y_true * y_pred, axis=axes
    )

    true_sum = tf.reduce_sum(
        y_true, axis=axes
    )

    pred_sum = tf.reduce_sum(
        y_pred, axis=axes
    )

    dice = (2.0 * intersection + smooth) / (
        true_sum + pred_sum + smooth
    )

    has_tumour = true_sum > 0

    tumour_dice = tf.boolean_mask(
        dice, has_tumour
    )

    return tf.cond(
        tf.size(tumour_dice) > 0,
        lambda: 1.0 - tf.reduce_mean(tumour_dice),
        lambda: tf.constant(0.0, dtype=tf.float32)
    )

# ============================================================
# Combined BCE + Dice loss
# ============================================================
def combined_loss(y_true, y_pred):
    bce = keras.losses.binary_crossentropy(
        y_true, y_pred
    )

    d_loss = dice_loss(y_true, y_pred)

    return tf.reduce_mean(bce) + d_loss

# ============================================================
# Compile model
# ============================================================


brain_model.compile(

    optimizer=keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    ),

    loss=combined_loss,

    metrics=[
        keras.metrics.BinaryAccuracy(
            name="binary_accuracy"
        ),

        keras.metrics.Precision(
            name="precision"
        ),

        keras.metrics.Recall(
            name="recall"
        ),

        dice_coef,
        binary_dice_coef
    ]
)


# ============================================================
# Save best model
# ============================================================

best_model_path = os.path.join(
    MODEL_DIRECTORY,
    "unet_brain_model_best.keras"
)


# ============================================================
# Callbacks
# ============================================================

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=best_model_path,
    monitor="val_dice_coef",
    mode="max",
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)


early_stopping_callback = keras.callbacks.EarlyStopping(
    monitor="val_dice_coef",
    mode="max",
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
    monitor="val_dice_coef",
    mode="max",
    factor=0.5,
    patience=10,
    min_lr=1e-6,
    verbose=1
)

# ============================================================
# Training history
# ============================================================

history_path = os.path.join(
    MODEL_DIRECTORY,
    "unet_training_history.json"
)



# ============================================================
# Train U-Net
# ============================================================

print("=" * 60)
print("Starting U-Net training")
print("=" * 60)


history = brain_model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    callbacks=[
        checkpoint_callback,
        reduce_lr_callback,
        early_stopping_callback
    ],

    verbose=1
)


# ============================================================
# Save final model
# ============================================================

final_model_path = os.path.join(
    MODEL_DIRECTORY,
    "unet_brain_model.keras"
)


# ============================================================
# Save training history
# ============================================================

with open(
    history_path,
    "w"
) as file:

    json.dump(
        history.history,
        file
    )


# ============================================================
# Load best model
# ============================================================

print("=" * 60)
print("Loading best validation model")
print("=" * 60)


best_model = keras.models.load_model(
    best_model_path,
    custom_objects={
        "dice_coef": dice_coef,
        "binary_dice_coef": binary_dice_coef,
        "combined_loss": combined_loss,
    }
)

best_model.save(
    final_model_path
)


# ============================================================
# Final test evaluation
# ============================================================

print("=" * 60)
print("Evaluating U-Net on test set")
print("=" * 60)


test_results = best_model.evaluate(
    test_dataset,
    verbose=1,
    return_dict=True
)


dice_value = float(
    test_results.get(
        "dice_coef",
        np.nan
    )
)

binary_dice_value = float(
    test_results.get(
        "binary_dice_coef",
        np.nan
    )
)

best_epoch_index = int(np.argmax(history.history["val_dice_coef"]))
best_epoch = best_epoch_index + 1
validation_accuracy = history.history.get(
    "val_binary_accuracy",
    history.history.get(
        "val_accuracy",
        [None]
    )
)

if validation_accuracy:
    validation_accuracy = float(
        validation_accuracy[best_epoch_index]
    )
else:
    validation_accuracy = None

print(
    f"Test dice: {dice_value:.4f}"
)

print(
    f"Binary test dice: {binary_dice_value:.4f}"
)

print(
    f"Validation accuracy: "
    f"{validation_accuracy:.4f}"
    if validation_accuracy is not None
    else "Validation accuracy: N/A"
)


# ============================================================
# Save test results
# ============================================================

test_results_path = os.path.join(
    MODEL_DIRECTORY,
    "test_results.json"
)


test_results_dict = {
    "dice": dice_value,
    "binary_dice": binary_dice_value,
    "validation_accuracy": validation_accuracy,
    "test_metrics": {
        name: float(value)
        for name, value in test_results.items()
    },
    "best_epoch": best_epoch,
    "best_validation_dice": float(history.history["val_dice_coef"][best_epoch_index]),
    "patient_split": "patient_split.json",
    "patient_counts": split_manifest["patient_counts"],
    "sample_counts": split_manifest["sample_counts"],
    "patient_leakage": split_manifest["patient_leakage"],
}


with open(
    test_results_path,
    "w"
) as file:

    json.dump(
        test_results_dict,
        file,
        indent=4
    )


# ============================================================
# Training complete
# ============================================================

print("=" * 60)

print(
    "U-Net training complete."
)

print(
    "Best model:",
    best_model_path
)

print(
    "Final model:",
    final_model_path
)

print(
    "Training history:",
    history_path
)

print(
    "Test results:",
    test_results_path
)

print("=" * 60)