"""Patient-level train, validation, and test splitting for BraTS H5 slices."""

import random
import re
from pathlib import Path


_SLICE_FILENAME = re.compile(r"^volume_(\d+)_slice_(\d+)\.h5$", re.IGNORECASE)


def patient_id_for_file(file_path):
    """Return the BraTS volume ID encoded in a slice filename."""
    filename = Path(file_path).name
    match = _SLICE_FILENAME.fullmatch(filename)
    if match is None:
        raise ValueError(
            f"Cannot determine patient volume from H5 filename {filename!r}; "
            "expected volume_<id>_slice_<index>.h5"
        )
    return f"volume_{int(match.group(1)):03d}"


def split_files_by_patient(files, seed):
 
    files_by_patient = {}
    for file_path in files:
        patient_id = patient_id_for_file(file_path)
        files_by_patient.setdefault(patient_id, []).append(str(file_path))

    patient_ids = sorted(files_by_patient)
    if len(patient_ids) < 3:
        raise ValueError(
            "Patient-level train/validation/test splitting requires at least "
            f"3 distinct volumes; found {len(patient_ids)}"
        )

    random.Random(seed).shuffle(patient_ids)
    train_end = max(1, min(int(0.8 * len(patient_ids)), len(patient_ids) - 2))
    validation_end = max(
        train_end + 1,
        min(int(0.9 * len(patient_ids)), len(patient_ids) - 1),
    )
    patient_groups = {
        "train": patient_ids[:train_end],
        "validation": patient_ids[train_end:validation_end],
        "test": patient_ids[validation_end:],
    }

    split_files = {
        split: sorted(
            file_path
            for patient_id in split_patients
            for file_path in files_by_patient[patient_id]
        )
        for split, split_patients in patient_groups.items()
    }

    patient_sets = [set(group) for group in patient_groups.values()]
    if any(
        patient_sets[left] & patient_sets[right]
        for left in range(len(patient_sets))
        for right in range(left + 1, len(patient_sets))
    ):
        raise RuntimeError("Patient leakage detected in generated data split")
    if sum(map(len, split_files.values())) != len(files):
        raise RuntimeError("Generated data split does not cover every input file")

    manifest = {
        "seed": seed,
        "strategy": "patient_volume_from_h5_filename",
        "patient_counts": {
            split: len(patient_groups[split]) for split in patient_groups
        },
        "sample_counts": {
            split: len(split_files[split]) for split in split_files
        },
        "splits": {
            split: {
                "patient_ids": patient_groups[split],
                "files": [Path(file_path).name for file_path in split_files[split]],
            }
            for split in patient_groups
        },
        "patient_leakage": False,
    }
    return split_files["train"], split_files["validation"], split_files["test"], manifest
