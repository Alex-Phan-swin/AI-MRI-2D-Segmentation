# AI MRI 2D Segmentation

This project trains a 2D brain tumor segmentation model from multi-modal MRI slice data using a U-Net style architecture. The implementation is designed for BraTS-style HDF5 samples where each slice contains a 4-channel MRI image and a segmentation mask. The pipeline includes preprocessing, augmentation, patient-level splitting, and model evaluation.

## Overview

The repository contains two main training workflows:

- `sage-maker/`: production training pipeline for AWS SageMaker
- `local-training/`: local notebook-based experiments and utilities
- `benchmark/`: historical Kaggle/U-Net benchmark models and comparison artifacts
- `outputs/`: generated predictions and outputs

The current trainer focuses on brain MRI segmentation from HDF5 slices and uses TensorFlow/Keras with patient-aware data splitting to avoid leakage across the same patient volume.

## Project goals

- Segment tumor regions from MR images in 2D slices
- Train on clinically realistic multi-modality MRI data
- Preserve patient-level separation between train/validation/test sets
- Support reproducible training with fixed seeds and saved split manifests
- Run locally or on Amazon SageMaker

## Repository structure

```text
AI-MRI-2D-Segmentation/
├── benchmark/                  # Legacy benchmark models and notebook
│   ├── README.md
│   ├── cnn_semantic.ipynb
│   ├── Models/
│   └── content/
├── local-training/             # Local training notebook and helpers
│   ├── train.ipynb
│   ├── split_utils.py
│   └── model/
├── outputs/                    # Generated predictions and outputs
├── sage-maker/                 # Primary training pipeline
│   ├── README.md
│   ├── requirements.txt
│   ├── split_utils.py
│   ├── train.py
│   ├── train_optimized.py
│   ├── launch_training.py
│   └── test_split_utils.py
├── .gitignore
└── README.md
```

## Data format

The training script expects HDF5 files in a flat directory with filenames following this pattern:

```text
volume_<patient-id>_slice_<slice-id>.h5
```

Each file should contain:

- `image`: MRI input array
- `mask`: segmentation target array

The project uses patient-aware splitting so all slices from the same volume remain in the same train, validation, or test split.

## Requirements

Python 3.10+ is recommended for the SageMaker configuration.

Core packages:

- TensorFlow 2.16.2
- h5py
- NumPy
- scikit-learn
- tqdm
- matplotlib

Install dependencies:

```bash
cd sage-maker
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Local training

Prepare a dataset folder named `data` inside `sage-maker/` or set the environment variable `SM_CHANNEL_TRAIN` to point at your dataset.

Then run:

```bash
cd sage-maker
python train.py
```

This script will:

- scan all `.h5` files in the training directory
- split them by patient
- create a deterministic patient split with seed `14`
- save the manifest to `patient_split.json`
- preprocess and augment MRI slices
- train the attention U-Net model
- save the best validation Dice checkpoint
- evaluate on the test split

## SageMaker training

The project includes a SageMaker launch script:

```bash
cd sage-maker
python launch_training.py
```

This script configures the standard training job for the AWS environment and runs the TensorFlow entry point defined in `train.py`.

## Validation and testing

The repository includes a split utility test suite:

```bash
cd sage-maker
python -m unittest discover -s . -p "test_split_utils.py" -v
```

## Model behavior

The training pipeline includes:

- per-modality intensity clipping and normalization
- background masking for non-brain voxels
- augmentation such as variance shifts, intensity scaling, and spatial mirroring
- attention-gated U-Net layers
- best-model checkpointing based on validation Dice score

## Outputs

Training outputs are written to the configured model directory and include:

- trained model checkpoints
- split metadata (`patient_split.json`)
- training logs and validation metrics
- prediction artifacts in `outputs/`

## Benchmark and historical models

Older experiments are stored in `benchmark/` and are intended for comparison and archival purposes. These include historical Kaggle/U-Net models and notebook-based evaluation workflows. See `benchmark/README.md` for more details.

## Notes

- The repository uses deterministic patient-based splitting, which helps avoid evaluation leakage.
- GPU training is intended for the configured SageMaker environment and is not required for every local workflow.
- This project is organized around MRI segmentation for brain tumor detection and model benchmarking rather than a generic image classification workflow.

## License

This project does not currently declare a license in the repository root. If you plan to share or publish the code, add an appropriate open-source license before distribution.
