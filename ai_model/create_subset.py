import h5py
import numpy as np

# =========================
# Paths
# =========================

input_file = "../dataset/raw/MIMIC-III_ppg_dataset.h5"
output_file = "../dataset/processed/ppg_subset.h5"

# =========================
# Subset Size
# =========================

subset_size = 10000

# =========================
# Create Subset
# =========================

with h5py.File(input_file, 'r') as infile:

    ppg = infile['ppg'][:subset_size]
    labels = infile['label'][:subset_size]
    subjects = infile['subject_idx'][:subset_size]

    with h5py.File(output_file, 'w') as outfile:

        outfile.create_dataset('ppg', data=ppg)
        outfile.create_dataset('label', data=labels)
        outfile.create_dataset('subject_idx', data=subjects)

print("\nSubset Created Successfully")
print(f"Saved to: {output_file}")
