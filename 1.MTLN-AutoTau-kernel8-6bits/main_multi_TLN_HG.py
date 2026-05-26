"""
### 1. Model Architecture Overview
The architecture is a Multimodal Threshold Logic Network (MTLN) designed for classifying OFDM signal modulations (BPSK, QPSK, QAM16, etc.) using hardware-friendly logic operations. It processes different intensity resolutions of the signal (bit planes) in parallel before fusing them.

#### Stage 1: Input & Binarization
- Binarization: Uses DistributiveThermometer to convert 2-channel input (Magnitude/Phase) into N bit planes (default 6).
- Multi-Branch Input: The data is split into N separate tensors. Each branch receives a 2-channel tensor representing a specific bit plane of the signal.

#### Stage 2: Per-Bit Branch Processing (make_branch)
The model initializes N parallel branches using MultiBranchFusion. Each branch consists of:
- Layer 1: SparseHierarchicalChannelLockedConv (Input: 2 channels -> Output: k channels).
- Activation: StepGateClippedSTE (a Straight-Through Estimator for binary logic).
- Pooling: MaxPool2d (2x2, stride 2).
- Layer 2: SparseChannelLockedConv (Output: 2k channels).
- Activation & Pooling: StepGateClippedSTE and MaxPool2d.

#### Stage 3: Global Fusion & Compression
- Concatenation: Outputs from all N branches are concatenated along the channel dimension.
- Fusion Layer 1: SparseChannelLockedConv (Output: 16 * k channels) followed by activation and pooling.
- Fusion Layer 2: SparseChannelLockedConv (Output: 32 * k channels) followed by activation and pooling.

#### Stage 4: Classification Head
- Flattening: Features are flattened into a large vector (e.g., 86,016 units).
- ThresholdLayer: A sparse linear layer where each output neuron connects to a fixed number (m=6 or 8) of inputs.
- GroupSum with Auto-Tau: Neurons are grouped by class. Their sums are scaled by an automatically calculated temperature tau = sqrt(group_size) (around 37 in the 6-bit configuration) to produce final logits.

### 2. Convolution Functions Analysis
The model relies on specialized sparse convolution layers designed to minimize the fan-in per neuron, making them efficient for FPGA/Verilog implementation.

- SparseHierarchicalChannelLockedConv: Implements hierarchical sparse connectivity. Used in Branch Layer 1 to bridge input channels to the initial feature set.
- SparseChannelLockedConv: A standard sparse convolution where each output channel is "locked" to a specific subset of input channels. Used in Branch Layer 2 and Fusion Layers.
- LogicMixedHierarchicalChannelLockedConv: Experimental branch using a 3rd learnable mixing gate to dynamically weigh binary feature interactions.

### 3. Key Parameters (Kernel-8 / 6-Bit Configuration)
- Num Bits: 6 (determines the number of parallel input branches).
- Num Kernels (k): 8 (base multiplier for channels).
- Kernel Size: 3x3 is used for all convolutions.
- Fan-in (m): Usually 6 or 8 (fixed connectivity per neuron).
- Total Layers: Approximately 29 weight-carrying or pooling layers.
"""
import argparse
import random
import subprocess
import time
import os

from huggingface_hub import snapshot_download

import numpy as np
from sympy import stats
import torch
import torch.nn as nn
from torch.nn import BatchNorm1d
import torchvision
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from collections import Counter

import os
try:
    from pipeline import run_full_logic_pipeline
except ImportError:
    run_full_logic_pipeline = None
from torch.optim.lr_scheduler import ReduceLROnPlateau
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    import mnist_dataset
except ImportError:
    pass
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)
from difflogic.threshold import ThresholdLayer, CustomSigmoid, CustomSigmoid2, GroupSum, StepGateSTE, StepGateClippedSTE, CustomSigmoid3
import difflogic.binarization as bin
from difflogic.connections import Conv, SparseChannelLockedConv, SparseDoubleChannelLockedConv, SparseHierarchicalChannelLockedConv, BranchFusion, SparseThresholdLinear, TwoGate45Then2ChannelLockedConv, LogicMixedHierarchicalChannelLockedConv

from datetime import datetime
print(datetime.now())



device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", device)

torch.set_num_threads(1)
torch.set_printoptions(threshold=float('inf'), linewidth=200)

BITS_TO_TORCH_FLOATING_POINT_TYPE = {
    16: torch.float16,
    32: torch.float32,
    64: torch.float64
}


class CustomModel(torch.nn.Module):
    """
    A wrapper around the model to enable verbose output during forward passes.
    """
    def __init__(self, model):
        super(CustomModel, self).__init__()
        self.model = model

    def forward(self, x, verbose=False):
        if verbose:
            print()
        y = self.model(x)
        if verbose:
            print()
        return y


def _format_shape(shape):
    if len(shape) == 0:
        return 'scalar'
    return 'x'.join(str(int(dim)) for dim in shape)


def _output_shape_to_str(output, drop_batch=True):
    if torch.is_tensor(output):
        shape = tuple(output.shape)
        if drop_batch and len(shape) >= 2:
            shape = shape[1:]
        return _format_shape(shape)

    if isinstance(output, (tuple, list)):
        tensor_parts = []
        for i, item in enumerate(output):
            if torch.is_tensor(item):
                tensor_parts.append(f'{i}:{_output_shape_to_str(item, drop_batch=drop_batch)}')
        return '[' + ', '.join(tensor_parts) + ']'

    return type(output).__name__


def print_layer_output_shapes(model, input_shape, input_dtype):
    base_model = model.model if isinstance(model, CustomModel) else model
    was_training = model.training
    model.eval()

    print('Layer output shapes (batch size = 1):')
    print("input:", " | ".join(
        _format_shape(s[1:]) for s in input_shape
    ))

    with torch.no_grad():
        if isinstance(input_shape[0], (tuple, list)):
            x = [torch.zeros(s, dtype=input_dtype, device=device) for s in input_shape]
        else:
            x = torch.zeros(input_shape, dtype=input_dtype, device=device)

        if isinstance(base_model, nn.Sequential):
            current = x
            for layer_idx, layer in enumerate(base_model):
                current = layer(current)
                shape_str = _output_shape_to_str(current, drop_batch=True)
                print(f'({layer_idx}) {layer.__class__.__name__}: {shape_str}')
        else:
            y = model(x)
            print(f'(output) {base_model.__class__.__name__}: {_output_shape_to_str(y, drop_batch=True)}')

    if was_training:
        model.train(True)


import psutil

def print_memory():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 3)
    print(f"Memory Usage: {mem:.2f} GB")


def load_dataset(args):
    num_bits = args.num_bits
    expected_channels = 2 * num_bits
    feature_wise = True

    if 'ofdm' not in args.dataset:
        raise NotImplementedError(f'The data set {args.dataset} is not supported! Please use ofdm.')

    class_names = ['BPSK', 'QPSK', 'QAM16', 'QAM64', 'QAM256', 'QAM1024']
    
    # Path to wireless data directory
    npy_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'wireless_data', '15dB'))

    # Check if data exists, if not, attempt to download it automatically.
    if not os.path.isdir(npy_dir) or not all(os.path.isdir(os.path.join(npy_dir, c)) for c in class_names):
        print(f"Data not found at {npy_dir}. Attempting to download from Hugging Face...")
        hf_token = args.hf_token if args.hf_token else os.getenv('HF_TOKEN')
        try:
            snapshot_download(
                repo_id='Sam10Man/Wireless',
                repo_type="dataset",
                local_dir=npy_dir,
                token=hf_token
            )
        except Exception as e:
             raise RuntimeError(f"Could not find or download data at {npy_dir}. Error: {e}")

    requested_cache_dir = os.path.abspath(args.preprocessed_cache_dir)
    # If the user didn't override the default, we might want to point it to wireless_data/preprocessed
    # But let's stay consistent with the provided arguments and cache logic.
    
    num_bits = args.num_bits
    expected_channels = 2 * num_bits
    feature_wise = True
    train_ratio = 0.7
    val_ratio = 0.1

    cache_config = {
        'dataset': args.dataset,
        'source_dir': npy_dir,
        'class_names': class_names,
        'train_ratio': train_ratio,
        'split_seed': args.seed,
        'thermometer_type': 'DistributiveThermometer',
        'num_bits': num_bits,
        'feature_wise': feature_wise,
    }
    cache_key = (
        f"{args.dataset}_distributive_{num_bits}bit_"
        f"featurewise_{int(feature_wise)}_seed_{args.seed}"
    )
    cache_dir = os.path.join(args.preprocessed_cache_dir, cache_key)
    train_cache_path = os.path.join(cache_dir, 'train.pt')
    val_cache_path = os.path.join(cache_dir, 'val.pt')
    meta_cache_path = os.path.join(cache_dir, 'meta.pt')
    test_cache_path = os.path.join(cache_dir, 'test.pt')

    if all(os.path.exists(path) for path in (train_cache_path, val_cache_path, meta_cache_path, test_cache_path)):
        print(f"Loading cached binarized OFDM data from {cache_dir}")
        train_cache = torch.load(train_cache_path, map_location='cpu')
        val_cache = torch.load(val_cache_path, map_location='cpu')
        test_cache = torch.load(test_cache_path, map_location='cpu')
        meta_cache = torch.load(meta_cache_path, map_location='cpu')

        # config check
        cached_config = meta_cache.get('cache_config', {})
        if cached_config.get('num_bits') != num_bits:
             raise RuntimeError(f"Cache mismatch. Requested {num_bits} bits, found {cached_config.get('num_bits')}")

        x_train_bin, y_train = train_cache['x'], train_cache['y']
        x_val_bin, y_val = val_cache['x'], val_cache['y']
        x_test_bin, y_test = test_cache['x'], test_cache['y']
    else:
        print("Loading OFDM datasets into memory for binarization...")
        train_files, train_labels = [], []
        val_files, val_labels = [], []
        test_files, test_labels = [], []
        rng = np.random.RandomState(args.seed)
        
        for class_idx, class_name in enumerate(class_names):
            class_path = os.path.join(npy_dir, class_name)
            files = sorted([os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith('.npy')])
            n_total = len(files)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            n_test = n_total - n_train - n_val
            
            indices = np.arange(n_total)
            rng.shuffle(indices)
            files = [files[i] for i in indices]
            
            train_files.extend(files[:n_train])
            train_labels.extend([class_idx] * n_train)
            val_files.extend(files[n_train:n_train+n_val])
            val_labels.extend([class_idx] * n_val)
            test_files.extend(files[n_train+n_val:])
            test_labels.extend([class_idx] * n_test)

        def load_npy_dataset(filepaths, labels):
            print(f"Loading {len(filepaths)} files...")
            print_memory()
            # Pre-allocate tensor to avoid multiple copies
            sample = np.load(filepaths[0])
            data = torch.empty((len(filepaths), *sample.shape), dtype=torch.float32)
            for i, f in enumerate(filepaths):
                try:
                    data[i] = torch.from_numpy(np.load(f))
                except Exception as e:
                    print(f"Error loading file {f} at index {i}: {e}")
                    raise e
                if i % 1000 == 0:
                    print(f"Loaded {i}/{len(filepaths)}...")
            
            labels = torch.tensor(labels, dtype=torch.long)
            print_memory()
            return data, labels

        x_train, y_train = load_npy_dataset(train_files, train_labels)
        x_val, y_val = load_npy_dataset(val_files, val_labels)
        x_test, y_test = load_npy_dataset(test_files, test_labels)

        print(f"Loaded train data: {x_train.shape}, val data: {x_val.shape}, test data: {x_test.shape}")
        
        print(f"Fitting {num_bits}-bit distributive thermometer on train data...")
        thermometer = bin.DistributiveThermometer(
            num_bits=num_bits,
            feature_wise=feature_wise
        ).fit(x_train)

        print("Binarizing train data...")
        # Process in chunks to avoid large intermediate tensors
        train_size = x_train.shape[0]
        x_train_bin = torch.empty((train_size, expected_channels, 256, 339), dtype=torch.bool)
        chunk_size_bin = 1000
        for i in range(0, train_size, chunk_size_bin):
            end = min(i + chunk_size_bin, train_size)
            bits = thermometer.binarize(x_train[i:end])
            x_train_bin[i:end] = bits.permute(0, 1, 4, 2, 3).reshape(-1, expected_channels, 256, 339)
            del bits
        del x_train
        
        print("Binarizing val data...")
        val_size = x_val.shape[0]
        x_val_bin = torch.empty((val_size, expected_channels, 256, 339), dtype=torch.bool)
        for i in range(0, val_size, chunk_size_bin):
            end = min(i + chunk_size_bin, val_size)
            bits = thermometer.binarize(x_val[i:end])
            x_val_bin[i:end] = bits.permute(0, 1, 4, 2, 3).reshape(-1, expected_channels, 256, 339)
            del bits
        del x_val
        
        print("Binarizing test data...")
        test_size = x_test.shape[0]
        x_test_bin = torch.empty((test_size, expected_channels, 256, 339), dtype=torch.bool)
        for i in range(0, test_size, chunk_size_bin):
            end = min(i + chunk_size_bin, test_size)
            bits = thermometer.binarize(x_test[i:end])
            x_test_bin[i:end] = bits.permute(0, 1, 4, 2, 3).reshape(-1, expected_channels, 256, 339)
            del bits
        del x_test
        
        print_memory()

        os.makedirs(cache_dir, exist_ok=True)
        torch.save({'x': x_train_bin, 'y': y_train}, train_cache_path)
        torch.save({'x': x_val_bin, 'y': y_val}, val_cache_path)
        torch.save({'x': x_test_bin, 'y': y_test}, test_cache_path)
        torch.save({'cache_config': cache_config}, meta_cache_path)
        print(f"Saved binarized OFDM cache to {cache_dir}")

    # =========================================================
    # Split num_bits branches
    # =========================================================
    train_branches = []
    val_branches = []
    test_branches = []

    print("Splitting bit branches and freeing full binarized tensors...")
    for bit_idx in range(num_bits):
        # -------------------------
        # TRAIN (Stacking returns a new tensor, slicing doesn't copy data until stacked)
        # -------------------------
        train_branches.append(torch.stack([
            x_train_bin[:, bit_idx],
            x_train_bin[:, bit_idx + num_bits]
        ], dim=1))

        # -------------------------
        # VALIDATION
        # -------------------------
        val_branches.append(torch.stack([
            x_val_bin[:, bit_idx],
            x_val_bin[:, bit_idx + num_bits]
        ], dim=1))

        # -------------------------
        # TEST
        # -------------------------
        test_branches.append(torch.stack([
            x_test_bin[:, bit_idx],
            x_test_bin[:, bit_idx + num_bits]
        ], dim=1))

    # Free the large combined tensors as they are no longer needed
    del x_train_bin, x_val_bin, x_test_bin
    print_memory()

    # =========================================================
    # Debug prints
    # =========================================================
    print("Number of branches:", len(train_branches))
    for i in range(num_bits):
        print(f"Branch {i} shape:", train_branches[i].shape)

    # =========================================================
    # Datasets
    # =========================================================
    train_dataset = torch.utils.data.TensorDataset(*train_branches, y_train)
    val_dataset = torch.utils.data.TensorDataset(*val_branches, y_val)
    test_dataset = torch.utils.data.TensorDataset(*test_branches, y_test)

    # =========================================================
    # DataLoaders
    # =========================================================
    map_train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        pin_memory=False, drop_last=True, num_workers=0
    )
    map_val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        pin_memory=False, drop_last=False, num_workers=0
    )
    map_test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        pin_memory=False, drop_last=False, num_workers=0
    )

    return map_train_loader, map_val_loader, map_test_loader

def input_dim_of_dataset(dataset, num_bits):
    """
    Return the input dimension for the specified dataset.
    """
    return {
        'ofdm': (2 * num_bits) * 256 * 339
    }[dataset]


def num_classes_of_dataset(dataset):
    """
    Return the number of classes for the specified dataset.
    """
    return {
        'ofdm': 6
    }[dataset]

# more important bit planes receive more kernels
# less important bit planes receive fewer kernels
def make_logic_mixed_branch(in_ch, k, weight=1):
    """
    Experimental branch using LogicMixedHierarchicalChannelLockedConv
    which includes a 3rd learnable mixing gate.
    """
    c1 = k * weight
    c2 = 2 * c1

    return torch.nn.Sequential(
        LogicMixedHierarchicalChannelLockedConv(
            in_channels=in_ch,
            out_channels=c1,
            kernel_size=3,
            stride=1,
            padding=1
        ),
        StepGateClippedSTE(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        LogicMixedHierarchicalChannelLockedConv(
            in_channels=c1,
            out_channels=c2,
            kernel_size=3,
            stride=1,
            padding=1
        ),
        StepGateClippedSTE(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )

# more important bit planes receive more kernels
# less important bit planes receive fewer kernels
def make_branch(in_ch, k, weight=1):

    c1 = k * weight
    c2 = 2 * c1

    return torch.nn.Sequential(

        SparseHierarchicalChannelLockedConv(
            in_channels=in_ch,
            out_channels=c1,
            kernel_size=3,
            stride=1,
            padding=1
        ),
        StepGateClippedSTE(),
        nn.MaxPool2d(kernel_size=2,stride=2),


        SparseChannelLockedConv(
            in_channels=c1,
            out_channels=c2,
            kernel_size=3,
            stride=1,
            padding=1
        ),
        StepGateClippedSTE(),
        nn.MaxPool2d(kernel_size=2,stride=2),
    )
class MultiBranchFusion(nn.Module):
    def __init__(self, branches):
        super().__init__()
        self.branches = nn.ModuleList(branches)   # IMPORTANT FIX

    def forward(self, xs):
        feats = []
        for branch, x in zip(self.branches, xs):
            feats.append(branch(x))
        return torch.cat(feats, dim=1)

def get_model(args, sample_loader):

    in_dim = input_dim_of_dataset(args.dataset, args.num_bits)
    class_count = num_classes_of_dataset(args.dataset)

    logic_layers = []
    arch = args.architecture
    k = args.num_kernels
    l = args.num_layers
    m = args.num_active

    if arch == 'threshold_connected':
        llkw = dict(grad_factor=args.grad_factor)

        if args.dataset == 'ofdm':
            # =========================================================
            # 15 BIT BRANCHES
            # =========================================================
            in_ch = 2  # magnitude + phase

            # Using the new logic mixing architecture
            """branches = torch.nn.ModuleList([
                make_logic_mixed_branch(in_ch, k).to(device)
                for _ in range(args.num_bits)
            ])"""
            branches = torch.nn.ModuleList([
                make_branch(in_ch, k).to(device)
                for _ in range(args.num_bits)
            ])

            # =========================================================
            # FUSION MODULE
            # =========================================================
            fusion_module = MultiBranchFusion(branches).to(device)

            # =========================================================
            # SPATIAL + CHANNEL COMPRESSION
            # =========================================================
            fusion_in_ch = args.num_bits * (2 * k )
            fusion_conv1 = SparseChannelLockedConv(
                in_channels= fusion_in_ch,
                out_channels=16 * k,
                kernel_size=3,
                stride=1,
                padding=1
            ).to(device)
            fusion_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            fusion_act1 = StepGateClippedSTE()

            # =========================================================
            # CROSS-FEATURE INTERACTION
            # =========================================================
            fusion_conv2 = SparseChannelLockedConv(
                in_channels=16 * k,
                out_channels=32 * k,
                kernel_size=3,
                stride=1,
                padding=1
            ).to(device)
            fusion_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            fusion_act2 = StepGateClippedSTE()

            # =========================================================
            # SHAPE INFERENCE PIPELINE
            # =========================================================
            def forward_features(x):
                x = fusion_module(x)
                x = fusion_conv1(x)
                x = fusion_act1(x)
                x = fusion_pool1(x)

                x = fusion_conv2(x)
                x = fusion_act2(x)
                x = fusion_pool2(x)

                return x

            sample_batch = next(iter(map_train_loader))
            *sample_branches, _ = sample_batch
            sample_branches = [b[:1].to(device) for b in sample_branches]

            fusion_module.eval()
            fusion_conv1.eval()
            fusion_conv2.eval()

            with torch.no_grad():
                out = forward_features(sample_branches)

            flatten_dim = out.flatten(1).shape[1]
            print("flatten_dim:", flatten_dim)

            # =========================================================
            # THRESHOLD LAYER
            # =========================================================
            compression_ratio = 0.1
            n = int(flatten_dim * compression_ratio)
            n = n - (n % class_count)

            threshold_layer = ThresholdLayer(
                in_dim=flatten_dim,
                out_dim=n,
                num_active=m,
                **llkw
            )

            # =========================================================
            # FINAL PIPELINE
            # =========================================================
            logic_layers = [
                fusion_module,
                fusion_conv1,
                fusion_act1,
                fusion_pool1,

                fusion_conv2,
                fusion_act2,
                fusion_pool2,

                torch.nn.Flatten(),
                threshold_layer,
                StepGateClippedSTE()
            ]

            group_size = n // class_count
            auto_tau = int(np.sqrt(group_size))
            print("------------>auto_tau:", auto_tau)

            model = torch.nn.Sequential(
                *logic_layers,
                GroupSum(class_count, auto_tau)
            )

            model = CustomModel(model).to(device)
    else:
        raise NotImplementedError(arch)

    # =========================================================
    # unchanged parts
    # =========================================================
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    if arch == 'threshold_connected':
        conv_layers = [
            layer for layer in model.modules()
            if isinstance(layer, (
                Conv,
                SparseChannelLockedConv,
                SparseHierarchicalChannelLockedConv,
                LogicMixedHierarchicalChannelLockedConv,
                TwoGate45Then2ChannelLockedConv,
            ))
        ]

        threshold_layers = [
            layer for layer in model.modules()
            if isinstance(layer, ThresholdLayer)
        ]

        conv_num_neurons = sum(layer.out_channels for layer in conv_layers)
        threshold_num_neurons = sum(layer.num_neurons for layer in threshold_layers)
        total_num_neurons = conv_num_neurons + threshold_num_neurons
        print(f'total_num_neurons={total_num_neurons}')
        total_num_weights = count_parameters(model)
        print(f'total_num_weights={total_num_weights}')

    model = model.to(device)

    print("model is on:", next(model.parameters()).device)
    print(model)

    input_dtype = next(model.parameters()).dtype

    if args.dataset == 'ofdm':
        print_layer_output_shapes(
            model,
            input_shape=[
                (1, 2, 256, 339)
                for _ in range(args.num_bits)
            ],
            input_dtype=input_dtype
        )

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.75, 0.90), weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.001, min_lr=1e-6)

    return model, loss_fn, optimizer, scheduler

def train_one_epoch(model, loader, loss_fn, optimizer, training_bit_count, epoch=None, log_interval=0):
    model.train(True)

    total_loss = 0.0
    total_correct = 0
    total_n = 0
    num_batches = len(loader)
    epoch_start_time = time.time()

    with open("log.txt", 'a') as log_file:  # Open the log file once

        for batch_idx, (*x_branches, y) in enumerate(loader, start=1):

            x_branches = [
                xb.to(
                    BITS_TO_TORCH_FLOATING_POINT_TYPE[training_bit_count]
                ).to(device, non_blocking=True)
                for xb in x_branches
            ]
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x_branches)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            bs = y.shape[0]
            total_loss += loss.item() * bs
            total_correct += (logits.argmax(-1) == y).sum().item()
            total_n += bs

            if log_interval > 0 and (batch_idx % log_interval == 0 or batch_idx == num_batches):
                elapsed = time.time() - epoch_start_time
                samples_per_sec = total_n / max(elapsed, 1e-9)
                prefix = f"[Epoch {epoch}] " if epoch is not None else ""
                log_line = (
                    f"{prefix}batch {batch_idx}/{num_batches} "
                    f"loss={total_loss / total_n:.6f} "
                    f"acc={total_correct / total_n:.4f} "
                    f"samples/s={samples_per_sec:.2f}"
                )
                print(log_line, flush=True)
                log_file.write(log_line + "\n")  # Save to file

        # Log final epoch statistics **after the loop ends**
        final_stats = f"Epoch {epoch} finished: total_loss={total_loss:.6f}, total_correct={total_correct}"
        print(final_stats, flush=True)
        log_file.write(final_stats + "\n")

    return total_loss / total_n, total_correct / total_n


def eval_acc(model, loader, training_bit_count):

    was_training = model.training
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for *x_branches, y in loader:
            x_branches = [
                xb.to(
                    BITS_TO_TORCH_FLOATING_POINT_TYPE[training_bit_count]
                ).to(device, non_blocking=True)
                for xb in x_branches
            ]

            y = y.to(device, non_blocking=True)

            pred = model(x_branches).argmax(-1)

            correct += (pred == y).sum().item()

            total += y.numel()

    if was_training:
        model.train(True)

    return correct / total


def eval_loss(model, loader, loss_fn, training_bit_count):

    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_n = 0

    with torch.no_grad():

        for *x_branches, y in loader:
            x_branches = [
                xb.to(
                    BITS_TO_TORCH_FLOATING_POINT_TYPE[training_bit_count]
                ).to(device, non_blocking=True)
                for xb in x_branches
            ]

            y = y.to(device, non_blocking=True)

            logits = model(x_branches)

            loss = loss_fn(logits, y)

            bs = y.shape[0]

            total_loss += loss.item() * bs
            total_n += bs

    if was_training:
        model.train(True)

    return total_loss / total_n


def eval_confusion_matrix(model, loader, training_bit_count, num_classes):

    was_training = model.training
    model.eval()

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    total_samples = 0

    with torch.no_grad():

        for *x_branches, y in loader:

            x_branches = [
                xb.to(
                    BITS_TO_TORCH_FLOATING_POINT_TYPE[training_bit_count]
                ).to(device, non_blocking=True)
                for xb in x_branches
            ]

            y = y.to(device, non_blocking=True)

            pred = model(x_branches).argmax(-1)

            y_np = y.detach().cpu().numpy()
            pred_np = pred.detach().cpu().numpy()

            np.add.at(cm, (y_np, pred_np), 1)

            total_samples += y_np.size

    if was_training:
        model.train(True)

    return cm, total_samples

def save_confusion_matrix_plot(cm, class_names, save_path, title="Confusion Matrix", values_format='d'):
    if plt is None:
        print("matplotlib is not installed; skipping confusion matrix plotting.")
        return False

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticks(ticks)
    ax.set_yticklabels(class_names)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title(title)

    threshold = (cm.max() / 2.0) if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm[i, j] > threshold else 'black'
            if values_format == 'd':
                text_value = f"{int(cm[i, j])}"
            else:
                text_value = format(cm[i, j], values_format)
            ax.text(j, i, text_value, ha='center', va='center', color=color)

    fig.tight_layout()
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return True


def run_logic_export_pipeline(best_ckpt_path, artifact_root, dataset_name):
    if run_full_logic_pipeline is None:
        raise RuntimeError("pipeline.py could not be imported, so --run-logic-pipeline cannot be used.")
    import tempfile

    with tempfile.TemporaryDirectory(prefix="logic_export_") as temp_root:
        csv_dir = os.path.join(temp_root, "csvfiles")
        truth_table_dir = os.path.join(temp_root, "truthtable")
        out_dir = os.path.join(temp_root, "out")
        verilog_dir = os.path.join(temp_root, "verilog")
        for directory in (csv_dir, truth_table_dir, out_dir, verilog_dir):
            os.makedirs(directory, exist_ok=True)

        artifact_tag = os.path.splitext(os.path.basename(best_ckpt_path))[0]
        print(f"[LOGIC] running pipeline for artifact {artifact_tag}", flush=True)
        run_full_logic_pipeline(
            dataset_name=dataset_name,
            model_weights_path=best_ckpt_path,
            csv_dir=csv_dir,
            truth_table_dir=truth_table_dir,
            out_dir=out_dir,
            artifact_tag=artifact_tag,
        )

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        resetfile_script = os.path.join(repo_root, "extras", "resetfile_generator.py")
        wrapper_script = os.path.join(repo_root, "extras", "neuron.py")

        subprocess.run(
            [sys.executable, resetfile_script, artifact_tag, "--expr-dir", out_dir, "--out-dir", verilog_dir],
            check=True,
        )
        subprocess.run(
            [sys.executable, wrapper_script, artifact_tag, "--expr-dir", out_dir, "--out-dir", verilog_dir],
            check=True,
        )

        print(f"[LOGIC] temporary outputs cleaned: {temp_root}", flush=True)


if __name__ == '__main__':
    default_n_bit = 6
    """
    Main function to set up and run the training and evaluation of the model based on command-line arguments.
    It handles argument parsing, dataset loading, model creation, training loop, and final evaluation.
    """
    parser = argparse.ArgumentParser(description='Train logic gate network on the various datasets.')
    parser.add_argument('-eid', '--experiment_id', type=int, default=None)
    parser.add_argument('--dataset', type=str, default='ofdm', choices=['ofdm'], required=False, help='the dataset to use')
    parser.add_argument('--tau', '-t', type=str, default='Auto-calculated', help='the softmax temperature tau')
    parser.add_argument('--seed', '-s', type=int, default=2017, help='seed (default: 0)')
    parser.add_argument('--batch-size', '-bs', type=int, default=32, help='batch size (default: 128)')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--training-bit-count', '-c', type=int, default=32, help='training bit count (default: 32)')
    parser.add_argument('--num-bits', '-nb', type=int, default=6, help='thermometer bit count for binarization (default: 3)')

    parser.add_argument('--implementation', type=str, default=device, choices=[device, 'python'],
                        help='`cuda` is the fast CUDA implementation and `python` is simpler but much slower '
                        'implementation intended for helping with the understanding.')

    parser.add_argument('--num-epochs', '-ne', type=int, default=100)
    parser.add_argument('--log-interval', type=int, default=0, help='print batch progress every N batches (0 disables)')
    parser.add_argument('--architecture', '-a', type=str, default='threshold_connected')
    parser.add_argument('--num_kernels', '-k', default=8, type=int)
    parser.add_argument('--num_layers', '-l', type=int)
    parser.add_argument('--num_active', '-m', type=int, default=6)
    parser.add_argument('--grad-factor', type=float, default=2.)
    parser.add_argument('--save-dir', type=str, default='weights3/chckpts', help='directory to save checkpoints')
    parser.add_argument('--save-best-name', type=str, default=None, help='override best checkpoint filename')
    parser.add_argument(
        '--run-logic-pipeline',
        action='store_true',
        help='after training, run checkpoint export -> csv -> truth table -> minimized expressions -> verilog on the best checkpoint',
    )
    parser.add_argument('--preprocessed-cache-dir',
        type=str,
        default=os.path.join(os.path.dirname(__file__), './../'),
        help='directory used to store and reuse cached binarized datasets',
    )
    parser.add_argument('--hf_token', type=str, default=None, help='Hugging Face API token')

    args = parser.parse_args()
    if args.num_bits < 1:
        parser.error('--num-bits must be >= 1')

    # Print the command used to run the script
    import sys
    import os
    script_name = os.path.basename(sys.argv[0])
    # Build the command string
    command_str = (
        f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 {script_name} "
        f"--dataset {args.dataset} "
        f"-bs {args.batch_size} "
        f"-nb {args.num_bits} "
        f"-ne {args.num_epochs} "
        f"-k {args.num_kernels} "
        f"-t {args.tau}"
    )
    if args.hf_token:
        command_str += f" --hf_token {args.hf_token}"
    
    print("\nCommand used to run this script:")
    print(command_str)

    print(vars(args))
    print(f"Using num_bits={args.num_bits} -> input channels={2 * args.num_bits}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    map_train_loader, map_validation_loader, map_test_loader = load_dataset(args)
    model, loss_fn, optim, scheduler = get_model(args, map_train_loader)

    os.makedirs(args.save_dir, exist_ok=True)

    artifact_root = os.path.dirname(args.save_dir) if os.path.basename(os.path.normpath(args.save_dir)) == 'chckpts' else args.save_dir
    ckpt_dir = args.save_dir
    cm_dir = os.path.join(artifact_root, 'confusionmatrices')
    training_dir = os.path.join(artifact_root, 'trainingstats')
    os.makedirs(cm_dir, exist_ok=True)
    os.makedirs(training_dir, exist_ok=True)

    best_val_acc = float("-inf")
    best_test_acc = float("nan")
    best_ckpt_path = args.save_best_name
    if best_ckpt_path is None:
        best_ckpt_path = os.path.join(
            ckpt_dir,
            f"best_{args.dataset}_arch={args.architecture}_k={args.num_kernels}_m={args.num_active}_nb={args.num_bits}_seed={args.seed}.pt"
        )
    train_losses, val_losses, test_losses = [], [], []
    train_accs,  val_accs,  test_accs  = [], [], []
    prev_lr = optim.param_groups[0]["lr"]

    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, map_train_loader, loss_fn, optim, args.training_bit_count, epoch=epoch, log_interval=args.log_interval
        )

        val_loss = eval_loss(model, map_validation_loader, loss_fn, args.training_bit_count)
        val_acc  = eval_acc(model, map_validation_loader, args.training_bit_count)
        test_loss = eval_loss(model, map_test_loader, loss_fn, args.training_bit_count)
        test_acc = eval_acc(model, map_test_loader, args.training_bit_count)

        train_losses.append(train_loss); val_losses.append(val_loss); test_losses.append(test_loss)
        train_accs.append(train_acc);    val_accs.append(val_acc);    test_accs.append(test_acc)

        scheduler.step(val_loss)
        lr_now = optim.param_groups[0]["lr"]
        lr_changed = (lr_now != prev_lr)
        if lr_changed:
            print(f"[LR] {prev_lr:.6g} -> {lr_now:.6g}")
            prev_lr = lr_now

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_test_acc = test_acc
            print(f"[CKPT] saved best at epoch {epoch} -> {best_ckpt_path}", flush=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_test_acc": best_test_acc,
                    "best_val_acc": best_val_acc,
                    "val_loss": val_loss,
                    "args": vars(args),
                },
                best_ckpt_path,
            )

        current_lr = optim.param_groups[0]["lr"]
        print({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test acc": test_acc,
            "lr": lr_now,
        }, flush=True)

    ckpt = torch.load(best_ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {})
    ckpt_num_bits = ckpt_args.get("num_bits", 3)
    if ckpt_num_bits != args.num_bits:
        raise RuntimeError(
            f"Checkpoint num_bits mismatch: checkpoint has num_bits={ckpt_num_bits}, "
            f"but current run uses num_bits={args.num_bits}."
        )
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss_best = eval_loss(model, map_test_loader, loss_fn, args.training_bit_count)
    test_acc_best  = eval_acc(model, map_test_loader, args.training_bit_count)
    class_names = ['BPSK', 'QPSK', 'QAM16', 'QAM64', 'QAM256', 'QAM1024']
    test_cm_best, test_cm_total = eval_confusion_matrix(
        model,
        map_test_loader,
        args.training_bit_count,
        num_classes=len(class_names),
    )

    cm_dir = os.path.abspath(cm_dir)
    cm_stem = (
        f"confusion_matrix_{args.dataset}_k{args.num_kernels}_"
        f"t{args.tau}_bs{args.batch_size}_nb{args.num_bits}_seed{args.seed}"
    )
    cm_path = os.path.join(cm_dir, f"{cm_stem}.png")

    os.makedirs(cm_dir, exist_ok=True)

    cm_saved = save_confusion_matrix_plot(
        test_cm_best,
        class_names,
        cm_path,
        title='Confusion Matrix (Best Checkpoint)',
    )

    log_file = "results.txt"

    with open(log_file, "w") as f:
        f.write(f"\nBest test acc: {best_test_acc * 100:.2f}%\n")
        f.write(f"Best-checkpoint test loss: {test_loss_best:.6f}, test acc: {test_acc_best * 100:.2f}%\n")
        f.write(f"Best checkpoint saved at: {best_ckpt_path}\n")
        f.write("Best-checkpoint confusion matrix:\n")
        f.write(str(test_cm_best) + "\n")
        f.write(f"Confusion matrix samples: {test_cm_total}\n")
    print(f"Results saved to: {log_file}")

    print(f"\nBest test acc: {best_test_acc*100:.2f}%")
    print(f"Best-checkpoint test loss: {test_loss_best:.6f}, test acc: {test_acc_best*100:.2f}%")
    print(f"Best checkpoint saved at: {best_ckpt_path}")
    print("Best-checkpoint confusion matrix:")
    print(test_cm_best)
    print(f"Confusion matrix samples: {test_cm_total}")
    if cm_saved:
        print(f"Confusion matrix plot saved at: {cm_path}")

    epochs = list(range(1, len(train_losses) + 1))

    if plt is None:
        print("matplotlib is not installed; skipping training curve plotting.")
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        ax1.plot(epochs, train_losses, label="Train Loss")
        ax1.plot(epochs, val_losses,   label="Val Loss")
        ax1.plot(epochs, test_losses,  label="Test Loss")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss vs Epoch")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(epochs, train_accs, label="Train Acc")
        ax2.plot(epochs, val_accs,   label="Val Acc")
        ax2.plot(epochs, test_accs,  label="Test Acc")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy vs Epoch")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        training_curve_name = f"training_k{args.num_kernels}_t{args.tau}_b{args.batch_size}_nb{args.num_bits}.png"
        plt.savefig(os.path.join(training_dir, training_curve_name), dpi=200, bbox_inches="tight")
        # plt.show()

    if args.run_logic_pipeline:
        run_logic_export_pipeline(best_ckpt_path, artifact_root, args.dataset)
