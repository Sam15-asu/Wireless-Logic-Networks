# Error Log: MTLN Script Execution

This file tracks the errors encountered while setting up and running the MTLN script and the steps taken to resolve them.

## [2026-05-24] Resolved: Environment Missing Packages
- **Errors**: `ModuleNotFoundError: No module named 'psutil'`, `ModuleNotFoundError: No module named 'numpy'`.
- **Resolution**: Installed missing packages in the virtual environment.

## [2026-05-24] Resolved: OOM during Binarization (EXIT 137)
- **Problem**: Binarizing 36,000 samples into 6-bit distributive thermometer encoding caused "Killed" error.
- **Root Cause**: Processing the entire dataset at once created a massive memory spike.
- **Fix**:
    1. **Stochastic Fitting**: Thresholds are now calculated based on a random 2000-sample subset.
    2. **Chunked Binarization**: Data is binarized in 1000-sample blocks using a pre-allocated boolean tensor.
    3. **Branch Caching**: Added code to save the 6 resulting bit-planes to disk (`ofdm_distributive_6bit_featurewise_1_seed_2017`). Loading now takes seconds instead of minutes.

## [2026-05-24] Resolved: CLI Parameter Ambiguity
- **Error**: `invalid int value: 'i'` for `-li 10`.
- **Cause**: `-li` was matched as `-l` (num_layers) and `i`.
- **Fix**: Changed to `--log-interval 10`.

## [2026-05-24] Resolved: DataLoader Memory Overhead (EXIT 137)
- **Problem**: Training script was "Killed" immediately after loading the model.
- **Root Cause**: Memory overhead from `num_workers=4` and `pin_memory=True` when handling large tensors on CPU.
- **Fix**: Modified `DataLoader` in `main_multi_TLN_sampad.py` to use `num_workers=0` and `pin_memory=False`.

## [2026-05-24] Current Status: Training
- **Action**: Run training with BS=32 on CPU.
- **Goal**: Complete at least 1 epoch to verify stability.
- **Monitoring**: `tail -f sam.txt`


