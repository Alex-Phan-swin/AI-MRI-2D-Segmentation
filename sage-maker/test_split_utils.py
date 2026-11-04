import unittest

from split_utils import patient_id_for_file, split_files_by_patient


class PatientSplitTests(unittest.TestCase):
    def setUp(self):
        self.files = [
            f"/data/volume_{patient:03d}_slice_{slice_id}.h5"
            for patient in range(10)
            for slice_id in range(patient % 3 + 1)
        ]

    def test_patient_id_comes_from_volume_not_slice(self):
        self.assertEqual(
            patient_id_for_file("/data/volume_7_slice_12.h5"),
            "volume_007",
        )

    def test_split_is_deterministic_disjoint_and_complete(self):
        first = split_files_by_patient(self.files, seed=14)
        second = split_files_by_patient(self.files, seed=14)

        self.assertEqual(first, second)
        train, validation, test, manifest = first
        split_sets = [
            set(manifest["splits"][name]["patient_ids"])
            for name in ("train", "validation", "test")
        ]
        self.assertFalse(split_sets[0] & split_sets[1])
        self.assertFalse(split_sets[0] & split_sets[2])
        self.assertFalse(split_sets[1] & split_sets[2])
        self.assertEqual(set().union(*split_sets), {f"volume_{i:03d}" for i in range(10)})
        self.assertEqual(len(train) + len(validation) + len(test), len(self.files))
        self.assertEqual(manifest["patient_leakage"], False)

        for split_files, patient_ids in zip(
            (train, validation, test), split_sets
        ):
            self.assertTrue(
                all(patient_id_for_file(file_path) in patient_ids for file_path in split_files)
            )

    def test_rejects_unrecognized_filenames(self):
        with self.assertRaisesRegex(ValueError, "expected volume_"):
            split_files_by_patient(
                ["/data/unrecognized.h5", "/data/volume_001_slice_0.h5",
                 "/data/volume_002_slice_0.h5"],
                seed=14,
            )

    def test_requires_three_distinct_patients(self):
        with self.assertRaisesRegex(ValueError, "at least 3 distinct volumes"):
            split_files_by_patient(
                ["/data/volume_001_slice_0.h5", "/data/volume_001_slice_1.h5"],
                seed=14,
            )


if __name__ == "__main__":
    unittest.main()
