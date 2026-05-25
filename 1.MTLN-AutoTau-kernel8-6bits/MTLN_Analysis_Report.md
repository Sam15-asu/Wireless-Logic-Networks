# MTLN-AutoTau-kernel8-6bits: Model Architecture and Analysis

This document provides a detailed explanation of the **Multimodal Threshold Logic Network (MTLN)** with hierarchical convolutions, specifically the version configured with kernel size 8 (effectively via branching/fusion) and 6-bit thermometer binarization.

## 1. Project Overview
The project implements a hardware-friendly logic-based neural network for **OFDM (Orthogonal Frequency Division Multiplexing)** signal classification. The model is designed to be easily exportable to Verilog (HDL) by using discrete logic operations (implemented via Threshold Logic) instead of standard floating-point arithmetic.

## 2. Model Architecture
The architecture follows a hierarchical structure: **Per-Bit Processing** followed by **Multimodal Fusion** and finally **Global Classification**.

### 2.1 Input Data & Binarization
- **Dataset**: OFDM signals (WiFi-15dB).
- **File Format**: The script expects a NumPy compressed archive (**`.npz`**) file.
- **Data Path**: The default path used in the script is `../../../1.WirelessTopic/datasets/wifi-15db-N-2x225x339.npz`.
- **Contents**: The `.npz` file must contain the following arrays:
  - `x_train`, `y_train`
  - `x_val`, `y_val`
  - `x_test`, `y_test`
- **Data Shape**: The input features (`x`) typically have a shape representing (Samples, Channels, Height, Width), e.g., `(8400, 2, 256, 339)` where channels are Magnitude and Phase.
- **Binarization**: Uses a **Distributive Thermometer Encoding** (`bin.DistributiveThermometer`). 
  - The input is expanded into $N_{bits}=6$ planes.
  - Since there are 2 antennas/channels and 6 bits per sample, the total input channels become $2 \times 6 = 12$.
  - **Feeding the Model**: The data is split into 6 separate tensors (one for each thermometer bit). Each tensor has a shape of `(Batch, 2, 256, 339)`. These are passed to the model as a list of branches, which the `MultiBranchFusion` module processes in parallel.

### 2.2 Hierarchical Branching (Multi-Branch Fusion)
The model processes each bit plane independently initially using a `MultiBranchFusion` module.
- **6 Parallel Branches**: One for each thermometer bit.
- **Detailed Structure of each branch (`make_branch`)**:
  - **Layer 1: SparseHierarchicalChannelLockedConv**
    - Input: 2 channels (Magnitude + Phase)
    - Output: $k$ channels (8 in this version)
    - Kernel Size: $3 \times 3$, Stride: 1, Padding: 1
    - Followed by `StepGateClippedSTE` activation.
  - **Pooling 1: MaxPool2d**
    - Kernel: $2 \times 2$, Stride: 2
    - Resolution: $256 \times 339 \to 128 \times 169$
  - **Layer 2: SparseChannelLockedConv**
    - Input: $k$ channels
    - Output: $2k$ channels (16 in this version)
    - Kernel Size: $3 \times 3$, Stride: 1, Padding: 1
    - Followed by `StepGateClippedSTE` activation.
  - **Pooling 2: MaxPool2d**
    - Kernel: $2 \times 2$, Stride: 2
    - Resolution: $128 \times 169 \to 64 \times 84$
- **Branch Output**: Each of the 6 branches outputs $16 \times 64 \times 84$ features.
- **Fusion**: The outputs of all 6 branches are concatenated along the channel dimension. Total channels = $6 \times 16 = 96$ channels.

### 2.3 Spatial & Channel Compression (Fusion Layers)
After fusion, the model applies two more stages of sparse convolutions to capture cross-bit interactions:
- **Fusion Layer 1: SparseChannelLockedConv**
  - Input: 96 channels (from 6 branches)
  - Output: $16 \times k = 128$ channels
  - Kernel Size: $3 \times 3$, Stride: 1, Padding: 1
  - Followed by `StepGateClippedSTE` and **MaxPool2d** ($2 \times 2$, Stride: 2).
  - Resolution: $64 \times 84 \to 32 \times 42$.
- **Fusion Layer 2: SparseChannelLockedConv**
  - Input: 128 channels
  - Output: $32 \times k = 256$ channels
  - Kernel Size: $3 \times 3$, Stride: 1, Padding: 1
  - Followed by `StepGateClippedSTE` and **MaxPool2d** ($2 \times 2$, Stride: 2).
  - Resolution: $32 \times 42 \to 16 \times 21$.

### 2.4 Layer Summary Table
| Stage | Layer Type | Count | Operations |
| :--- | :--- | :---: | :--- |
| **Branches** | Conv2D (Sparse) | 12 | 2 per branch $\times$ 6 branches |
| | MaxPool2D | 12 | 2 per branch $\times$ 6 branches |
| **Fusion** | Conv2D (Sparse) | 2 | Sequential cross-bit interaction |
| | MaxPool2D | 2 | Sequential downsampling |
| **Classifier**| Threshold (FC) | 1 | Global feature processing |
| **Total** | | **29** | Total weight-carrying/pooling layers |

### 2.4 Threshold Layer & Classification
- **Flattening**: The $256 \times 16 \times 21$ features are flattened into a vector of size **86,016**.
- **ThresholdLayer**: A sparse linear layer where each output neuron is connected to exactly $m=6$ inputs.
  - **Output Size ($n$)**: Calculated as $10\%$ of input dim (approx 8,598), aligned to the number of classes.
- **GroupSum**: The final layer sums groups of neurons to produce logits for the 6 classes (BPSK, QPSK, QAM16, QAM64, QAM256, QAM1024).
  - **Auto-Tau**: A temperature parameter ($\tau$) used for scaling the sums, automatically calculated as $\sqrt{group\_size}$. In this run, $\tau = 37$.

## 3. Dimensions and Neuron Count Analysis
Based on the `long.txt` log:
- **Input Channels**: 12 (6 bits $\times$ 2 components).
- **Flattened Dimension**: 86,016.
- **Total Neurons**: 9,078.
- **Total Weights**: 64,218.

| Layer | Type | Output Shape | Neurons |
| :--- | :--- | :--- | :--- |
| **Branch Layer 1** | SparseConv | $6 \times 8 \times 128 \times 169$ | 48 |
| **Branch Layer 2** | SparseConv | $6 \times 16 \times 64 \times 84$ | 96 |
| **Fusion Layer 1** | SparseConv | $128 \times 32 \times 42$ | 128 |
| **Fusion Layer 2** | SparseConv | $256 \times 16 \times 21$ | 256 |
| **Global FC** | Threshold | 8,598 | 8,598 |
| **Total** | | | **9,078** |

### Why these dimensions?
1. **Sparse Connectivity**: The model uses sparse connections (fan-in=6 or 8) to reduce the number of hardware logic gates required.
2. **Channel Locking**: `SparseChannelLockedConv` ensures that each output neuron only looks at a specific hardware-efficient subset of inputs.
3. **Hierarchy**: Processing bit planes separately first allows the model to learn features at different "resolution" levels of the signal intensity before combining them.

## 4. Kernel Architecture and Logic Convolution
The MTLN architecture replaces standard, dense CNN operations with **Sparse Channel-Locked Logic Convolutions**. These are designed for direct efficient translation into hardware logic gates.

### 4.1 Input Connectivity and Sparsity
Unlike a standard CNN kernel where every pixel in a $3 \times 3$ window is used ($9$ weights), the MTLN kernels use **Sparse Taps**:
- **Layer 1 (Hierarchical Branching)**: Each output neuron selects exactly **2 input channels**. Within each of those channels, it only picks **6 out of 9** available pixels (for a $3 \times 3$ kernel).
- **Layer 2 (Channel Locked)**: Each output neuron is assigned to exactly **1 input channel**. Within that channel's $3 \times 3$ patch, it picks **6 or 8 sparse taps**.
- **The Taps**: The choice of which pixels to tap into within the $k \times k$ window is **fixed and random** upon initialization. This mimics a hard-wired circuit rather than a flexible soft-processor.

### 4.2 How it differs from Standard CNNs
| Feature | Standard CNN | MTLN (Logic Conv) |
| :--- | :--- | :--- |
| **Channel Mixing** | **Dense**: Every output channel sums over ALL input channels. | **Locked**: Each output neuron only looks at 1 or 2 specific input channels. |
| **Kernel Density** | **Dense**: All $K \times K$ weights are learnable/used. | **Sparse**: Only a subset of pixels (e.g., 6 of 9) are connected. |
| **Operation** | Floating-point Multiply-Accumulate (MAC). | Thresholded Sum-and-Compare: $\sum w_i x_i - \theta$. |
| **Activations** | Continuous Non-linearities (ReLU, GELU). | Discrete Step Gates: $\{0, 1\}$ outputs. |
| **Cross-Feature Interaction** | Happens at every layer. | **Delayed**: Explicitly decoupled until the Fusion Stage. |

### 4.3 Logic-Mixed Hierarchical Branching ($k=8$)
The model uses an advanced **`LogicMixedHierarchicalChannelLockedConv`** in the first layer of each branch. This is where the initial "Multimodal Mixing" occurs between the Magnitude and Phase components of the binarized signal.

**The Internal Logic Structure:**
1.  **Level 1 (Feature Extraction)**:
    *   **Partial Gate A (Magnitude)**: Connects to **6 sparse pixels** from the Magnitude $3 \times 3$ window. It produces a raw sum $z_{mag}$, which is then **binarized** via STE into a single bit.
    *   **Partial Gate B (Phase)**: Connects to **6 sparse pixels** from the Phase $3 \times 3$ window. It produces a raw sum $z_{phase}$, which is then **binarized** via STE into a single bit.
2.  **Level 2 (Logic Mixing Gate)**:
    *   A **3rd learnable 2-input gate** takes the bits from Magnitude and Phase as inputs.
    *   This gate has its own **2 weights ($w_1, w_2$) and 1 threshold ($\theta$)**.
    *   It learns the optimal logic relationship (e.g., AND, OR, XOR) between the Mag and Phase features to decide if the kernel should fire.

**Where is it used?**
*   **Hierarchical Branching (Layer 1)**: **YES**. It is used as the entry point for each of the 6 bit-branches to integrate Mag and Phase information into a unified logic representation.
*   **Intra-Branch Convolution (Layer 2)**: **NO**. This stage uses `SparseChannelLockedConv`, which focuses on refining features within the already-mixed channels.
*   **Fusion Stage (Cross-Bit Interaction)**: **NO**. The fusion layers use `SparseChannelLockedConv` to compress the 96 channels ($6 \text{ branches} \times 16 \text{ channels}$) into higher-level abstractions.

### 4.4 Operation Principle
Instead of learning a general-purpose filter, each "kernel" in this network acts as a **learnable logic gate**. By limiting the fan-in (number of inputs) and locking the channels, the model forces the representation to be highly compressed and structured. The "mixing" of information from different bit-planes or signal components is strictly controlled and only occurs in the hierarchical fusion modules, preventing the exponential growth of complexity typical of large CNNs.

## 4. Training Procedure
1. **Loss Function**: `CrossEntropyLoss`.
2. **Optimizer**: `AdamW` with learning rate 0.01, specific betas (0.75, 0.90), and weight decay 1e-4.
3. **Scheduler**: `ReduceLROnPlateau` (factor 0.1, patience 5).
4. **STE (Straight-Through Estimator)**: Since the model uses hard step functions (0 or 1), gradients are bypassed using STE to allow backpropagation.
5. **Epochs**: 100 epochs, saving the best checkpoint based on validation accuracy.

## 5. Hardware Export (Logic Pipeline)
The project includes a pipeline (`run_logic_export_pipeline`) that:
1. Extracts weights into CSV files.
2. Generates Truth Tables for the logic gates.
3. Performs logic minimization.
4. Exports the entire model as **Verilog (.v)** for FPGA/ASIC implementation.
