
import os
from pathlib import Path

import boto3
import sagemaker

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

CONFIG_PATH = PROJECT_ROOT / ".sagemaker-code-config"

if not CONFIG_PATH.exists():
    CONFIG_PATH.write_text(
        '{"sagemakerProjectId":"local-dev"}',
        encoding="utf-8"
    )

from sagemaker.tensorflow import TensorFlow


# ============================================================
# Configuration
# ============================================================

AWS_REGION = "ap-southeast-2"

ROLE_ARN = (

)

BUCKET = 

DATA_PATH = f"s3://{BUCKET}/data/"
OUTPUT_PATH = f"s3://{BUCKET}/models/"


# ============================================================
# SageMaker session
# ============================================================

boto_session = boto3.Session(
    region_name=AWS_REGION
)

session = sagemaker.Session(
    boto_session=boto_session,
    default_bucket=BUCKET
)


# ============================================================
# Create TensorFlow estimator
# ============================================================

estimator = TensorFlow(
    entry_point="train.py",
    source_dir=".",
    role=ROLE_ARN,

    framework_version="2.16.2",
    py_version="py310",

    instance_type="ml.g5.xlarge",
    instance_count=1,

    output_path=OUTPUT_PATH,

    max_run=24 * 60 * 60,

    sagemaker_session=session,

    environment={
        "PYTHONUNBUFFERED": "1"
    },
)

# ============================================================
# Start training
# ============================================================

print("=" * 60)
print("Starting SageMaker TensorFlow training job")
print("=" * 60)

print("Dataset:", DATA_PATH)
print("Output:", OUTPUT_PATH)
print("Instance:", "ml.g5.xlarge")

print("=" * 60)


estimator.fit({
    "train": DATA_PATH
})


# ============================================================
# Training complete
# ============================================================

print("=" * 60)
print("Training job completed.")
print("Model artifacts:")
print(estimator.model_data)
print("=" * 60)