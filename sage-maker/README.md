# BraTS training on Amazon SageMaker

`launch_training.py` submits `train.py` as the SageMaker TensorFlow entry point.
It uses the configured S3 training prefix and model output prefix, the
`ml.g5.xlarge` instance, TensorFlow 2.16.2, and Python 3.10. Run it from this
directory in an AWS-enabled Python environment with SageMaker permissions:

```bash
python launch_training.py
```

The training channel must contain flat HDF5 slice files named
`volume_<patient-id>_slice_<slice-id>.h5`, each with `image` and `mask` datasets.
`train.py` and `train_optimized.py` both use the shared
[`split_utils.py`](./split_utils.py) helper to split by volume ID. All slices
from one volume stay in a single train, validation, or test split.

The split is deterministic for seed 14. `patient_split.json` records the exact
patient and slice membership and is written to SageMaker's model-artifact
directory. `train.py` saves the best validation-Dice checkpoint and evaluates
the test split once. `train_optimized.py` selects hyperparameters by validation
Dice, then evaluates the untouched test split once for the selected run.

To run locally for data/split checks, place the HDF5 files in `./data`; the
trainer uses that path when the SageMaker `SM_CHANNEL_TRAIN` variable is absent.
Local runs need compatible TensorFlow, h5py, and NumPy packages. GPU training
is intended for the configured SageMaker instance, not the repository's local
Python environment.

The standard-library split tests can be run with:

```bash
python -m unittest discover -s . -p "test_split_utils.py" -v
```
