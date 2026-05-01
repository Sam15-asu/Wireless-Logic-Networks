# OFDM Logic Network Flow

This note explains `experiments2/main_baseline_conv_conf.py`, the custom layers it uses, how the project flow differs from a normal CNN

## Main Script Overview

`experiments2/main_baseline_conv_conf.py` trains a 6-class OFDM modulation classifier for:

- `BPSK`
- `QPSK`
- `QAM16`
- `QAM64`
- `QAM256`
- `QAM1024`

The raw samples are `.npy` files shaped `(2, 256, 339)`. The two channels are magnitude and phase, created by the conversion utility in `data3/datamodulator.py` or the root `datamodulator.py`.

The important flow is:

1. Load OFDM `.npy` samples from `data3/15dB/<class_name>`.
2. Split each class into train/validation/test using `70% / 10% / 20%`.
3. Fit a `DistributiveThermometer` on the training tensors.
4. Convert each continuous feature into `num_bits` binary thermometer bits.
5. Reshape the data from `(N, 2, 256, 339, num_bits)` into `(N, 2 * num_bits, 256, 339)`.
6. Train a sparse threshold-convolution network.
7. Save the checkpoint with best validation accuracy.
8. Evaluate the best checkpoint on the test set and save a confusion matrix.
9. Optionally run the logic export pipeline, which turns learned threshold gates into Boolean expressions and Verilog-oriented artifacts.

For the file named with `nb=15`, every sample has `2 * 15 = 30` binary input channels.

## Data Binarization

The binarization comes from `experiments2/difflogic/binarization.py`.

`DistributiveThermometer(num_bits=15, feature_wise=True)` computes data-dependent quantile thresholds for each feature position. For each scalar feature, it creates 15 thresholds and outputs 15 binary comparisons:

```text
bit_j = feature_value > threshold_j
```

Because the original sample has 2 channels, after thermometer encoding the model sees:

```text
30 x 256 x 339
```

This is already a major departure from a normal CNN. The network is not learning from continuous magnitude/phase directly; it learns from a binary expanded representation.

## Network Architecture

For `architecture='threshold_connected'`, `get_model()` builds this sequence:

| Sequential index | Layer | Output shape for `k=16`, `nb=15` |
|---:|---|---|
| input | binary OFDM tensor | `30 x 256 x 339` |
| 0 | `SparseChannelLockedConv(30 -> 16)` | `16 x 256 x 339` |
| 1 | `StepGateClippedSTE` | `16 x 256 x 339` |
| 2 | `MaxPool2d(2)` | `16 x 128 x 169` |
| 3 | `SparseChannelLockedConv(16 -> 32)` | `32 x 128 x 169` |
| 4 | `StepGateClippedSTE` | `32 x 128 x 169` |
| 5 | `MaxPool2d(2)` | `32 x 64 x 84` |
| 6 | `SparseChannelLockedConv(32 -> 64)` | `64 x 64 x 84` |
| 7 | `StepGateClippedSTE` | `64 x 64 x 84` |
| 8 | `MaxPool2d(2)` | `64 x 32 x 42` |
| 9 | `SparseChannelLockedConv(64 -> 128)` | `128 x 32 x 42` |
| 10 | `StepGateClippedSTE` | `128 x 32 x 42` |
| 11 | `MaxPool2d(2)` | `128 x 16 x 21` |
| 12 | `SparseChannelLockedConv(128 -> 256)` | `256 x 16 x 21` |
| 13 | `StepGateClippedSTE` | `256 x 16 x 21` |
| 14 | `MaxPool2d(2)` | `256 x 8 x 10` |
| 15 | `Flatten` | `20480` |
| 16 | `ThresholdLayer(20480 -> 7998, num_active=6)` | `7998` |
| 17 | `StepGateClippedSTE` | `7998` |
| final | `GroupSum(6, tau)` | `6` logits |

The final `GroupSum` expects the `7998` binary units to be divisible by 6 classes. It reshapes them into 6 groups and sums each group:

```text
logit_class_c = sum(units assigned to class c) / tau
```

For `n=7998`, each class receives:

```text
7998 / 6 = 1333
```

voting-like binary units.

## Custom Sparse Logic Layers

### `SparseChannelLockedConv`

Defined in `experiments2/difflogic/connections.py`.

This is the most important layer. It behaves like a convolution structurally, but its computation is a sparse threshold gate, not a normal learned dense kernel.

For each output channel:

1. The layer chooses exactly one input channel.
2. It unfolds a local `3 x 3` patch from that channel.
3. It chooses only `fan_in=6` positions from the 9 possible pixels in that one channel.
4. It learns 6 weights and one threshold.
5. It outputs a preactivation:

```text
z = w1*x[idx1] + w2*x[idx2] + ... + w6*x[idx6] - theta
```

Then `StepGateClippedSTE` turns that value into a binary output:

```text
output = 1 if z >= 0 else 0
```

During backpropagation, the step function uses a straight-through estimator with clipped gradients, so the forward pass is hard binary but training still receives approximate gradients.

### `ThresholdLayer`

Defined in `experiments2/difflogic/threshold.py`.

This is the sparse final classifier layer. Each of the 7998 neurons connects to only 6 inputs out of the flattened 20480 features:

```text
z_j = sum_i weight[j, i] * x[idx[j, i]] - bias[j]
```

Again, `StepGateClippedSTE` binarizes the output.

### `GroupSum`

Defined in `experiments2/difflogic/threshold.py`.

This layer groups binary neurons into class groups and sums them. It is closer to a learned binary voting system than to a dense linear classifier.

## How This Differs From a Normal CNN

A normal CNN usually has dense learned kernels. For example, each output channel in a standard `Conv2d` uses all input channels and all positions in the kernel window:

```text
out_channel_j = sum over all input channels and all kernel positions
```

This project is different:

- Inputs are thermometer-binarized before the model sees them.
- Convolutional layers are sparse threshold gates.
- Each output channel is locked to one input channel, not all input channels.
- Each `3 x 3` kernel uses only 6 selected taps, not all 9 positions and not all channels.
- Activations are hard binary steps in the forward pass.
- Training uses STE gradients to make hard binary gates trainable.
- The final classifier is not a dense linear head; it is 7998 sparse threshold gates followed by class-wise summation.
- The trained model can be converted into Boolean expressions and hardware-style logic.

So the architecture has CNN-like spatial hierarchy and pooling, but the learned computation is much closer to a sparse Boolean/threshold circuit.


## Summary

The project is a CNN-shaped sparse threshold-logic classifier for OFDM modulation recognition. It uses convolution-like spatial processing and pooling, but the actual learned units are small six-input threshold gates with binary outputs. The minimized expression file is a symbolic export of those gates: each line maps one trained sparse gate to a minimized Boolean formula, plus the source indices and the list of inputs that must be inverted to preserve the original signed-weight threshold behavior.

## Command To Run

Run from the experiments2 directory:

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 main_baseline_conv_conf.py --dataset ofdm -bs 128 -nb 15 -ne 40 -k 32 -n 7998 -t 100.0
