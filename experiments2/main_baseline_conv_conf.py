import argparse
import random
import subprocess
import time

import numpy as np
from sympy import stats
import torch
import torch.nn as nn
from torch.nn import BatchNorm1d
import torchvision
from torchvision import datasets, transforms
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
from difflogic.connections import Conv, SparseChannelLockedConv, SparseThresholdLinear, TwoGate45Then2ChannelLockedConv

from datetime import datetime
print(datetime.now())


torch.set_num_threads(1)
torch.set_printoptions(threshold=float('inf'), linewidth=200)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", device)
print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available(), "| built-with CUDA:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

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
    print(f'input: {_format_shape(input_shape[1:])}')

    with torch.no_grad():
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


def _find_data3_root():
    script_dir = os.path.dirname(__file__)
    candidates = [
        os.path.join(script_dir, '..', 'data3'),
        os.path.join(script_dir, '..', '..', 'data3'),
        os.path.join(os.getcwd(), 'data3'),
        os.path.join(os.getcwd(), '..', 'data3'),
    ]

    checked = []
    for candidate in candidates:
        data_root = os.path.abspath(candidate)
        if data_root in checked:
            continue
        checked.append(data_root)
        if os.path.isdir(os.path.join(data_root, '15dB')):
            return data_root

    raise RuntimeError(
        "Could not find OFDM data3/15dB directory. Checked: "
        + ", ".join(checked)
    )


def load_dataset(args):
    """
    Load the OFDM dataset, prepare (binarize) the data, and return DataLoaders
    for training, validation, and testing.
    """
    if 'ofdm' not in args.dataset:
        raise NotImplementedError(f'The data set {args.dataset} is not supported! Please use ofdm.')

    class_names = ['BPSK', 'QPSK', 'QAM16', 'QAM64', 'QAM256', 'QAM1024']
    data3_root = _find_data3_root()
    npy_dir = os.path.join(data3_root, '15dB')

    default_cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data3', 'preprocessed'))
    requested_cache_dir = os.path.abspath(args.preprocessed_cache_dir)
    if requested_cache_dir == default_cache_dir:
        args.preprocessed_cache_dir = os.path.join(data3_root, 'preprocessed')

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

    if all(os.path.exists(path) for path in (train_cache_path, val_cache_path, meta_cache_path)):
        print(f"Loading cached binarized OFDM data from {cache_dir}")
        train_cache = torch.load(train_cache_path, map_location='cpu')
        val_cache = torch.load(val_cache_path, map_location='cpu')
        meta_cache = torch.load(meta_cache_path, map_location='cpu')
        test_cache_path = os.path.join(cache_dir, 'test.pt')
        if not os.path.exists(test_cache_path):
            raise RuntimeError(f"Missing test.pt in cache directory {cache_dir}. Please delete the cache and rerun.")
        test_cache = torch.load(test_cache_path, map_location='cpu')

        if meta_cache.get('cache_config') != cache_config:
            cached_config = meta_cache.get('cache_config', {})
            raise RuntimeError(
                f"Preprocessed cache metadata mismatch for {cache_dir}. "
                f"Requested num_bits={num_bits}, cached num_bits={cached_config.get('num_bits')}. "
                "Delete the cache directory to rebuild it."
            )

        x_train_bin, y_train = train_cache['x'], train_cache['y']
        x_val_bin, y_val = val_cache['x'], val_cache['y']
        x_test_bin, y_test = test_cache['x'], test_cache['y']
        if (
            x_train_bin.shape[1] != expected_channels or
            x_val_bin.shape[1] != expected_channels or
            x_test_bin.shape[1] != expected_channels
        ):
            raise RuntimeError(
                f"Cached channel mismatch for num_bits={num_bits}: "
                f"train channels={x_train_bin.shape[1]}, val channels={x_val_bin.shape[1]}, test channels={x_test_bin.shape[1]}, "
                f"expected channels={expected_channels}. Delete {cache_dir} and rerun."
            )
        print(f"Loaded cached train data: {x_train_bin.shape}, val data: {x_val_bin.shape}, test data: {x_test_bin.shape}")
    else:

        print("Loading OFDM datasets into memory for binarization...")
        # Split per class to guarantee exact test/val/train counts
        train_files, train_labels = [], []
        val_files, val_labels = [], []
        test_files, test_labels = [], []
        rng = np.random.RandomState(args.seed)
        test_samples_per_class = []
        for class_idx, class_name in enumerate(class_names):
            class_path = os.path.join(npy_dir, class_name)
            files = sorted([os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith('.npy')])
            n_total = len(files)
            # Use integer division for train and val, assign remainder to test for perfect balance
            n_train = int(n_total * 0.7)
            n_val = int(n_total * 0.1)
            n_test = n_total - n_train - n_val  # Ensures all samples are used
            indices = np.arange(n_total)
            rng.shuffle(indices)
            files = [files[i] for i in indices]
            train_files.extend(files[:n_train])
            train_labels.extend([class_idx] * n_train)
            val_files.extend(files[n_train:n_train+n_val])
            val_labels.extend([class_idx] * n_val)
            test_files.extend(files[n_train+n_val:n_train+n_val+n_test])
            test_labels.extend([class_idx] * n_test)
            test_samples_per_class.append((class_name, n_test))

        print("Test samples per class:")
        for cname, ntest in test_samples_per_class:
            print(f"  {cname}: {ntest} samples")

        # Shuffle the splits globally for randomness
        def shuffle_together(files, labels):
            idx = np.arange(len(files))
            rng.shuffle(idx)
            files = [files[i] for i in idx]
            labels = [labels[i] for i in idx]
            return files, labels
        train_files, train_labels = shuffle_together(train_files, train_labels)
        val_files, val_labels = shuffle_together(val_files, val_labels)
        test_files, test_labels = shuffle_together(test_files, test_labels)

        def load_npy_dataset(filepaths, labels):
            data = [np.load(f) for f in filepaths]
            data = np.stack(data)
            labels = np.array(labels)
            return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

        x_train, y_train = load_npy_dataset(train_files, train_labels)
        x_val, y_val = load_npy_dataset(val_files, val_labels)
        x_test, y_test = load_npy_dataset(test_files, test_labels)

        print(f"Loaded train data: {x_train.shape}, val data: {x_val.shape}, test data: {x_test.shape}")
        print(f"Fitting {num_bits}-bit distributive thermometer on train data...")
        thermometer = bin.DistributiveThermometer(num_bits=num_bits, feature_wise=feature_wise).fit(x_train)

        print("Binarizing train, val, and test data...")
        x_train_bits = thermometer.binarize(x_train)
        x_val_bits = thermometer.binarize(x_val)
        x_test_bits = thermometer.binarize(x_test)

        x_train_bin = x_train_bits.permute(0, 1, 4, 2, 3).reshape(x_train_bits.size(0), expected_channels, 256, 339)
        x_val_bin = x_val_bits.permute(0, 1, 4, 2, 3).reshape(x_val_bits.size(0), expected_channels, 256, 339)
        x_test_bin = x_test_bits.permute(0, 1, 4, 2, 3).reshape(x_test_bits.size(0), expected_channels, 256, 339)

        if x_train_bin.shape[1] != expected_channels or x_val_bin.shape[1] != expected_channels or x_test_bin.shape[1] != expected_channels:
            raise RuntimeError(
                f"Generated channel mismatch for num_bits={num_bits}: "
                f"train channels={x_train_bin.shape[1]}, val channels={x_val_bin.shape[1]}, test channels={x_test_bin.shape[1]}, "
                f"expected channels={expected_channels}."
            )

        os.makedirs(cache_dir, exist_ok=True)
        torch.save({'x': x_train_bin, 'y': y_train}, train_cache_path)
        torch.save({'x': x_val_bin, 'y': y_val}, val_cache_path)
        torch.save({'x': x_test_bin, 'y': y_test}, os.path.join(cache_dir, 'test.pt'))
        torch.save(
            {
                'cache_config': cache_config,
                'thresholds': thermometer.thresholds,
                'x_train_shape': tuple(x_train.shape),
                'x_val_shape': tuple(x_val.shape),
                'x_test_shape': tuple(x_test.shape),
                'x_train_bin_shape': tuple(x_train_bin.shape),
                'x_val_bin_shape': tuple(x_val_bin.shape),
                'x_test_bin_shape': tuple(x_test_bin.shape),
            },
            meta_cache_path,
        )
        print(f"Saved binarized OFDM cache to {cache_dir}")

    print(f"Final binarized train shape: {x_train_bin.shape}")

    train_dataset_bin = torch.utils.data.TensorDataset(x_train_bin, y_train)
    val_dataset_bin = torch.utils.data.TensorDataset(x_val_bin, y_val)
    test_dataset_bin = torch.utils.data.TensorDataset(x_test_bin, y_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset_bin, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, drop_last=True, num_workers=4
    )

    validation_loader = torch.utils.data.DataLoader(
        val_dataset_bin, batch_size=args.batch_size, shuffle=False,
        pin_memory=True, drop_last=False, num_workers=4
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset_bin, batch_size=args.batch_size, shuffle=False,
        pin_memory=True, drop_last=False, num_workers=4
    )

    return train_loader, validation_loader, test_loader


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


def get_model(args):
    """
    Create and return the model, loss function, optimizer, and scheduler based on the provided arguments.
    Model architecture is built according to the specified dataset and parameters.
    Loss function is CrossEntropyLoss.
    Optimizer is Adam with specified learning rate and betas.
    Scheduler is ReduceLROnPlateau monitoring validation loss.
    """
    in_dim = input_dim_of_dataset(args.dataset, args.num_bits)
    class_count = num_classes_of_dataset(args.dataset)

    logic_layers = []
    arch = args.architecture
    k = args.num_kernels
    n = args.num_neurons
    l = args.num_layers
    m = args.num_active
    total_num_neurons = 0

    if arch == 'threshold_connected':
        llkw = dict(grad_factor=args.grad_factor)

        if args.dataset == 'ofdm':
            in_ch = 2 * args.num_bits

            logic_layers.append(SparseChannelLockedConv(in_channels=in_ch, out_channels=k, kernel_size=3, stride=1, padding=1))
            logic_layers.append(StepGateClippedSTE())
            logic_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            logic_layers.append(SparseChannelLockedConv(in_channels=k, out_channels=2*k, kernel_size=3, stride=1, padding=1))
            logic_layers.append(StepGateClippedSTE())
            logic_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            logic_layers.append(SparseChannelLockedConv(in_channels=2*k, out_channels=4*k, kernel_size=3, stride=1, padding=1))
            logic_layers.append(StepGateClippedSTE())
            logic_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            logic_layers.append(SparseChannelLockedConv(in_channels=4*k, out_channels=8*k, kernel_size=3, stride=1, padding=1))
            logic_layers.append(StepGateClippedSTE())
            logic_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            logic_layers.append(SparseChannelLockedConv(in_channels=8*k, out_channels=16*k, kernel_size=3, stride=1, padding=1))
            logic_layers.append(StepGateClippedSTE())
            logic_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            logic_layers.append(torch.nn.Flatten())

            flatten_dim = (256 // 32) * (339 // 32) * (16 * k)
            #flatten_dim = 3 * 4 * (16 * k)

            logic_layers.append(ThresholdLayer(in_dim=flatten_dim, out_dim=n, num_active=m, **llkw))
            logic_layers.append(StepGateClippedSTE())


        model = torch.nn.Sequential(*logic_layers, GroupSum(class_count, args.tau))
        model = CustomModel(model)

    else:
        raise NotImplementedError(arch)

    def count_parameters(model):
        """
        Count the number of trainable parameters in the model.
        Args: model (torch.nn.Module): The model to count parameters for.
        Returns: Prints the number of parameters
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    if arch == 'threshold_connected':
        conv_layers = [
            layer for layer in logic_layers
            if isinstance(layer, (Conv, SparseChannelLockedConv, TwoGate45Then2ChannelLockedConv))
        ]
        threshold_layers = [layer for layer in logic_layers if isinstance(layer, ThresholdLayer)]
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
        print_layer_output_shapes(model, input_shape=(1, 2 * args.num_bits, 256, 339), input_dtype=input_dtype)
    print('\n')

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

    for batch_idx, (x, y) in enumerate(loader, start=1):
        x = x.to(BITS_TO_TORCH_FLOATING_POINT_TYPE[training_bit_count]).to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
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
            print(
                (
                    f"{prefix}batch {batch_idx}/{num_batches} "
                    f"loss={total_loss / total_n:.6f} "
                    f"acc={total_correct / total_n:.4f} "
                    f"samples/s={samples_per_sec:.2f}"
                ),
                flush=True,
            )

    return total_loss / total_n, total_correct / total_n


def eval_acc(model, loader, training_bit_count):
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(BITS_TO_TORCH_FLOATING_POINT_TYPE[training_bit_count]).to(device)
            y = y.to(device)
            pred = model(x).argmax(-1)
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
        for x, y in loader:
            x = x.to(BITS_TO_TORCH_FLOATING_POINT_TYPE[training_bit_count]).to(device)
            y = y.to(device)
            logits = model(x)
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
        for x, y in loader:
            x = x.to(BITS_TO_TORCH_FLOATING_POINT_TYPE[training_bit_count]).to(device)
            y = y.to(device)
            pred = model(x).argmax(-1)

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
    """
    Main function to set up and run the training and evaluation of the model based on command-line arguments.
    It handles argument parsing, dataset loading, model creation, training loop, and final evaluation.
    """
    parser = argparse.ArgumentParser(description='Train logic gate network on the various datasets.')
    parser.add_argument('-eid', '--experiment_id', type=int, default=None)
    parser.add_argument('--dataset', type=str, choices=[
        'ofdm'
    ], required=True, help='the dataset to use')
    parser.add_argument('--tau', '-t', type=float, default=10, help='the softmax temperature tau')
    parser.add_argument('--seed', '-s', type=int, default=0, help='seed (default: 0)')
    parser.add_argument('--batch-size', '-bs', type=int, default=128, help='batch size (default: 128)')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--training-bit-count', '-c', type=int, default=32, help='training bit count (default: 32)')
    parser.add_argument('--num-bits', '-nb', type=int, default=3, help='thermometer bit count for binarization (default: 3)')

    parser.add_argument('--implementation', type=str, default=device, choices=[device, 'python'],
                        help='`cuda` is the fast CUDA implementation and `python` is simpler but much slower '
                        'implementation intended for helping with the understanding.')

    parser.add_argument('--num-epochs', '-ne', type=int, default=30)
    parser.add_argument('--log-interval', type=int, default=0, help='print batch progress every N batches (0 disables)')
    parser.add_argument('--architecture', '-a', type=str, default='threshold_connected')
    parser.add_argument('--num_kernels', '-k', type=int)
    parser.add_argument('--num_neurons', '-n', type=int)
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
    parser.add_argument(
        '--preprocessed-cache-dir',
        type=str,
        default=os.path.join(os.path.dirname(__file__), '..', 'data3', 'preprocessed'),
        help='directory used to store and reuse cached binarized datasets',
    )


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
        f"-n {args.num_neurons} "
        f"-t {args.tau}"
    )
    print("\nCommand used to run this script:")
    print(command_str)

    print(vars(args))
    print(f"Using num_bits={args.num_bits} -> input channels={2 * args.num_bits}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_loader, validation_loader, test_loader = load_dataset(args)
    model, loss_fn, optim, scheduler = get_model(args)

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
            f"best_{args.dataset}_arch={args.architecture}_k={args.num_kernels}_n={args.num_neurons}_m={args.num_active}_nb={args.num_bits}_seed={args.seed}.pt"
        )
    train_losses, val_losses, test_losses = [], [], []
    train_accs,  val_accs,  test_accs  = [], [], []
    prev_lr = optim.param_groups[0]["lr"]

    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optim, args.training_bit_count, epoch=epoch, log_interval=args.log_interval
        )

        val_loss = eval_loss(model, validation_loader, loss_fn, args.training_bit_count)
        val_acc  = eval_acc(model, validation_loader, args.training_bit_count)
        test_loss = eval_loss(model, test_loader, loss_fn, args.training_bit_count)
        test_acc = eval_acc(model, test_loader, args.training_bit_count)

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

    test_loss_best = eval_loss(model, test_loader, loss_fn, args.training_bit_count)
    test_acc_best  = eval_acc(model, test_loader, args.training_bit_count)
    class_names = ['BPSK', 'QPSK', 'QAM16', 'QAM64', 'QAM256', 'QAM1024']
    test_cm_best, test_cm_total = eval_confusion_matrix(
        model,
        test_loader,
        args.training_bit_count,
        num_classes=len(class_names),
    )

    cm_dir = os.path.abspath(cm_dir)
    cm_stem = (
        f"confusion_matrix_{args.dataset}_k{args.num_kernels}_"
        f"n{args.num_neurons}_t{args.tau:g}_bs{args.batch_size}_nb{args.num_bits}_seed{args.seed}"
    )
    cm_path = os.path.join(cm_dir, f"{cm_stem}.png")

    os.makedirs(cm_dir, exist_ok=True)

    cm_saved = save_confusion_matrix_plot(
        test_cm_best,
        class_names,
        cm_path,
        title='Confusion Matrix (Best Checkpoint)',
    )

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
        training_curve_name = f"training_k{args.num_kernels}_n{args.num_neurons}_t{args.tau}_b{args.batch_size}_nb{args.num_bits}.png"
        plt.savefig(os.path.join(training_dir, training_curve_name), dpi=200, bbox_inches="tight")
        plt.show()

    if args.run_logic_pipeline:
        run_logic_export_pipeline(best_ckpt_path, artifact_root, args.dataset)
