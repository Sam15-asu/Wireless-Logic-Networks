# MTLN (Multi-branch Threshold Logic Network) Flow

This document explains the flow of the MTLN project for OFDM modulation classification.

## 1. Data Processing and Binarization
- **Input**: Complex OFDM signals (I/Q) stored as `.npy` files.
- **Loading**: The `load_npy_dataset` function gathers file paths for 36,000 samples across 6 modulation classes (BPSK, QPSK, QAM16, QAM64, QAM256, QAM1024).
- **Encoding**: Uses **Distributive Thermometer Encoding**.
  - Signals are converted to Magnitude and Phase.
  - Thresholds are calculated using a stochastic subset (2,000 samples) to avoid memory overflow.
  - Data is binarized into 6 "bit-branches".
- **Caching**: The binarized branches are saved to disk (`train_branches_ofdm_6.pt`, etc.) in `torch.bool` format to bypass the intensive binarization step in subsequent runs.

## 2. Model Architecture
- **Architecture Type**: `threshold_connected`.
- **Branches**:
  - Each of the 6 bits from the binarization goes into its own branch.
  - Each branch is a `ThresholdLayer` followed by activation.
- **Fusion**: `MultiBranchFusion` concatenates the outputs of all branches.
- **Dynamic Logic**: The core uses `ThresholdLayer` which implements logic gates using thresholds and Straight-Through Estimators (STE). This allows training discrete logic networks using backpropagation.
- **Auto-Tau**: The `tau` parameter (annealing temperature for the logic gates) is auto-calculated based on the number of bits and layers.

## 3. Training Process
- **Environment**: Optimized for CPU-only training with high memory availability (500GB+).
- **DataLoader**: Configured with `num_workers=0` and `pin_memory=False` to minimize memory overhead when handling the ~70GB dataset.
- **Loss Function**: `CrossEntropyLoss` for the 6-class modulation classification.
- **Optimizer**: Adam optimizer with a learning rate of 0.01.

## 4. Execution Flow
1. Check for cached bit-branches.
2. If not found, perform binarization in chunks and save to disk.
3. Initialize the `MultiBranchFusion` model.
4. Run the training loop for the specified number of epochs.
5. Log accuracy and loss per batch and epoch.
6. Evaluate on the validation and test sets.
