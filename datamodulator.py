# CSV to NPY Conversion + PyTorch Dataset for OFDM Modulation Data
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import glob
import os


def to_complex(s):
    """Convert a string like '1.5+0.3i' to a Python complex number."""
    return complex(s.replace(" ", "").replace("i", "j"))


def convert_csv_to_npy(base_dir, class_names, output_dir, shape=(256, 339)):
    """
    Convert all CSV files to .npy files with magnitude and phase channels.
    
    Each CSV file is converted to a numpy array of shape (2, H, W) where:
        - Channel 0: magnitude
        - Channel 1: phase
    
    Output directory structure mirrors the input:
        output_dir/
            QAM1024/
                sample_0000.npy
                sample_0001.npy
                ...
            QPSK/
                ...
    """
    os.makedirs(output_dir, exist_ok=True)

    for class_name in class_names:
        class_input_dir = os.path.join(base_dir, class_name)
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        csv_files = sorted(glob.glob(os.path.join(class_input_dir, '*.csv')))
        print(f"[{class_name}] Found {len(csv_files)} CSV files")

        for i, csv_path in enumerate(csv_files):
            df = pd.read_csv(csv_path, header=None, dtype=str)
            flat = df.values.flatten()
            data_complex = np.array([to_complex(s) for s in flat], dtype=np.complex64).reshape(shape)

            mag = np.abs(data_complex)
            phase = np.angle(data_complex)
            sample = np.stack((mag, phase), axis=0)  # shape: (2, 256, 339)

            # Use original filename with .npy extension
            base_name = os.path.splitext(os.path.basename(csv_path))[0]
            npy_path = os.path.join(class_output_dir, f"{base_name}.npy")
            np.save(npy_path, sample)

            if (i + 1) % 100 == 0 or (i + 1) == len(csv_files):
                print(f"  [{class_name}] Converted {i + 1}/{len(csv_files)}")

    print(f"\nAll conversions complete! Files saved to: {output_dir}")


class NPYModulationDataset(Dataset):
    """PyTorch Dataset that loads pre-converted .npy files."""
    def __init__(self, base_dir, class_names, split='train', train_ratio=0.8, random_seed=None):
        self.class_names = class_names
        self.class_to_label = {name: i for i, name in enumerate(class_names)}
        self.filepaths = []

        def _is_valid_npy(path):
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                return False
            try:
                _ = np.load(path, mmap_mode='r')
                return True
            except Exception:
                return False

        for class_name in class_names:
            class_path = os.path.join(base_dir, class_name)
            all_files = sorted(glob.glob(os.path.join(class_path, '*.npy')))

            if random_seed is not None:
                np.random.seed(random_seed)
                np.random.shuffle(all_files)

            num_train = int(len(all_files) * train_ratio)
            selected_files = (
                all_files[:num_train] if split == 'train' else all_files[num_train:]
            )

            for fname in selected_files:
                if _is_valid_npy(fname):
                    self.filepaths.append((fname, self.class_to_label[class_name]))

        if random_seed is not None:
            np.random.seed(random_seed + 1)
        np.random.shuffle(self.filepaths)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath, label = self.filepaths[idx]
        sample = np.load(filepath)  # shape: (2, 256, 339)
        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return sample, label


# ===== Configuration =====

# ===== New Conversion Function for Flat Folder =====
def convert_flat_csv_folder_to_npy(input_dir, output_dir, shape=(256, 339)):
    """
    Convert all CSV files in input_dir to .npy files in output_dir.
    Each .npy file will have shape (2, H, W) with magnitude and phase channels.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_files = sorted(glob.glob(os.path.join(input_dir, '*.csv')))
    print(f"[convert] Found {len(csv_files)} CSV files in {input_dir}")
    for i, csv_path in enumerate(csv_files):
        df = pd.read_csv(csv_path, header=None, dtype=str)
        flat = df.values.flatten()
        data_complex = np.array([to_complex(s) for s in flat], dtype=np.complex64).reshape(shape)
        mag = np.abs(data_complex)
        phase = np.angle(data_complex)
        sample = np.stack((mag, phase), axis=0)
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        npy_path = os.path.join(output_dir, f"{base_name}.npy")
        np.save(npy_path, sample)
        if (i + 1) % 100 == 0 or (i + 1) == len(csv_files):
            print(f"  Converted {i + 1}/{len(csv_files)}")
    print(f"All conversions complete for {input_dir} → {output_dir}")

# ===== Run Flat Folder Conversion for QAM16 and QAM64 =====
if __name__ == '__main__':
    # QAM1024 only
    QAM1024_input = '/workspace/sampad/wireless/data3/15dB/QAM1024/QAM1024'
    QAM1024_output = '/workspace/sampad/wireless/data3/15dB/QAM1024_npy'
    convert_flat_csv_folder_to_npy(QAM1024_input, QAM1024_output, shape=(256, 339))

    # Verification
    sample_files = glob.glob(os.path.join(QAM1024_output, '*.npy'))
    if sample_files:
        sample = np.load(sample_files[0])
        print(f"Sample from {QAM1024_output}: shape {sample.shape} (expected: (2, 256, 339))")
