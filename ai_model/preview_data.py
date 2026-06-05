import h5py
import numpy as np

file_path = "../dataset/raw/MIMIC-III_ppg_dataset.h5"

with h5py.File(file_path, 'r') as f:

    ppg = f['ppg']
    labels = f['label']
    subject_ids = f['subject_idx']

    print("\n===== DATASET INFO =====\n")

    print("PPG Shape:", ppg.shape)
    print("Label Shape:", labels.shape)
    print("Subject Shape:", subject_ids.shape)

    print("\n===== FIRST SAMPLE =====\n")

    print("PPG Sample:")
    print(ppg[0][:20])

    print("\nLabel:")
    print(labels[0])

    print("\nSubject ID:")
    print(subject_ids[0])
