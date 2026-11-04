# Historical Kaggle/U-Net benchmark

This directory contains the older binary U-Net experiments trained on the
Kaggle LGG image/mask dataset. It includes model checkpoints, training
histories, test summaries, and a semantic-model notebook.

`Models/model_to_use.txt` records the previous project selection:
`model_300_epoch_reducelr_earlystp` or `model_unet_sagemaker`. The artifacts
remain available for historical comparison. The current BraTS/SageMaker
trainer uses a different four-modality HDF5 dataset and TensorFlow model, so it
does not load these Kaggle checkpoints.

The Kaggle demo source and instructions for downloading its dataset are kept on
the [`legacy` branch](https://github.com/Mannat2004/ComputingTechnologyProjectBP45/tree/legacy/legacy/sprint4-kaggle-demo).

The model archive is saved with Keras 3.13.2. Run `cnn_semantic.ipynb` with
TensorFlow 2.16.1 or newer and Keras 3.13.2 or newer. In the notebook's Python
environment, install a compatible Keras version with:

```bash
python -m pip install --upgrade "keras>=3.13.2"
```

The model-loading cells check the archive metadata and report a clear version
error if the installed Keras is too old to deserialize it.
