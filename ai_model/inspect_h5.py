import h5py

# Path to dataset
file_path = "../dataset/raw/MIMIC-III_ppg_dataset.h5"

# Open dataset
with h5py.File(file_path, 'r') as f:

    print("\n===== HDF5 STRUCTURE =====\n")

    def print_structure(name, obj):
        print(name)

    f.visititems(print_structure)
