# Summary of Changes and Adaptations to `main_multi_TLN_sampad.py`

This document details the modifications made to the original MTLN script to support the `data3/15dB` dataset and improve training efficiency.

## 1. Dataset Loading Overhaul
The original script relied on a single `.npz` file containing pre-split training, validation, and test sets. The updated version implements a distributed loading mechanism:

- **Directory-Based Scanning**: The script now searches the `/workspace/data3/15dB/` directory, specifically looking into subfolders for each modulation class: `BPSK`, `QPSK`, `QAM16`, `QAM64`, `QAM256`, and `QAM1024`.
- **Dynamic File Discovery**: It automatically collects all `.npy` files from these directories, allowing it to handle any number of samples (e.g., the 6,000 samples per class provided).
- **Balanced Splitting**: A consistent Train/Validation/Test split (70%/10%/20%) is applied per class. This ensures every modulation scheme is equally represented in all subsets of the data.
- **Root Path Discovery**: Added the `_find_data3_root()` helper function to ensure the script can find the dataset whether it is run from the workspace root or the project folder.

## 2. Binarization Caching System
Binarizing thousands of high-resolution signal samples (256x339) into multiple bit-planes is computationally expensive. To solve this, a caching system was added:

- **Preprocessed Storage**: After the first binarization, the processed tensors and labels are saved as PyTorch files (`.pt`) in the `preprocessed/` directory.
- **Cache Keying**: The cache is uniquely identified by the dataset name, bit count, and random seed.
- **Fast Loading**: Subsequent runs check for these files and load them instantly, reducing the startup time from minutes to seconds.
- **Metadata Validation**: The script saves a `meta.pt` file to ensure the cached data matches the currently requested configuration (e.g., matching the number of bits).

## 3. Integration with Multi-Branch Architecture
While the data loading was changed, the specific **MTLN Branching Logic** was preserved and correctly mapped to the new data:

- **Bit-Plane Pairing**: After binarization, the 12 resulting channels (for 6-bit encoding) are still grouped such that the $i$-th bit of Magnitude and the $i$-th bit of Phase are paired together in **Branch $i$**.
- **Data Tensor Conversion**: The loaded NumPy files are converted to PyTorch tensors with the correct `float32` and `long` types required by the logic network layers.

## 4. Pipeline Compatibility
- **Return Type Fix**: The `load_dataset` function was updated to return the specific DataLoaders required by the rest of the existing training pipeline (`map_train_loader`, `map_val_loader`, `map_test_loader`).
- **Parallel Workers**: Enabled `num_workers=4` in the DataLoaders to ensure the CPU can keep up with the GPU when feeding the large binarized tensors.

## Summary Table of File Logic

| Feature | Original Script (`main_multi_TLN.py`) | Updated Script (`main_multi_TLN_sampad.py`) |
| :--- | :--- | :--- |
| **Data Source** | Single `.npz` file | Multiple folders/files in `data3/15dB` |
| **Startup Time**| Moderate (loading one large file) | **Fast** (via Caching) / Slow (first run) |
| **Preprocessing**| Always performed | **Cached** after first successful run |
| **Data Split** | Fixed in `.npz` | **Dynamic** 70/10/20 per class |
| **Architecture** | 6-bit Hierarchical MTLN | 6-bit Hierarchical MTLN (Preserved) |
