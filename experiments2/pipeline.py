#!/usr/bin/env python3
import os
import csv
import itertools
import re
from functools import reduce
from pathlib import Path
import torch
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# 1) Export effective weights from .pth → model_effective_params_<dataset>.csv
# ──────────────────────────────────────────────────────────────────────────────

def _load_checkpoint_state_dict(model_weights_path):
    """
    Accept either:
      - a raw state_dict saved directly with torch.save(model.state_dict(), ...), or
      - a richer checkpoint dict containing model_state_dict.
    """
    checkpoint = torch.load(model_weights_path, map_location='cpu')

    if isinstance(checkpoint, dict):
        model_state_dict = checkpoint.get("model_state_dict")
        if isinstance(model_state_dict, dict):
            return model_state_dict

        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint

    raise ValueError(
        f"Unsupported checkpoint format in {model_weights_path}. "
        "Expected a raw state_dict or a checkpoint containing 'model_state_dict'."
    )


def _resolve_artifact_tag(dataset_name, model_weights_path=None, artifact_tag=None):
    """
    Choose a stable suffix for all generated artifact filenames.

    Priority:
      1. explicit artifact_tag
      2. checkpoint stem (e.g. best_ofdm_arch=..._k=..._seed=...)
      3. dataset_name
    """
    if artifact_tag:
        return artifact_tag
    if model_weights_path:
        return Path(model_weights_path).stem
    return dataset_name


def _extract_sparse_layer_id(parameter_name):
    """
    Return the numeric Sequential index for sparse layers that were exported to CSV.
    Examples:
      model.0.effective_weight -> 0
      model.16.weight          -> 16
    """
    match = re.search(r"model\.(\d+)\.(?:effective_weight|weight)$", str(parameter_name).strip())
    if not match:
        return None
    return int(match.group(1))


def _pad_weight_values(values, width=6):
    values = [str(val) for val in values[:width]]
    while len(values) < width:
        values.append("")
    return values


def _write_sparse_rows(writer, parameter_name, weight_tensor, idx_tensor, bias_tensor):
    weight_np = weight_tensor.detach().cpu().numpy()
    idx_np = idx_tensor.detach().cpu().numpy()
    bias_np = bias_tensor.detach().cpu().numpy() if bias_tensor is not None else None

    for row_index, row in enumerate(weight_np):
        row_flat = row.flatten()
        idx_row = idx_np[row_index].flatten()
        bias_value = bias_np[row_index] if (bias_np is not None and row_index < bias_np.shape[0]) else ""
        writer.writerow([
            parameter_name,
            row_index,
            row_flat.shape,
            ",".join(map(str, idx_row.tolist()))
        ] + _pad_weight_values(row_flat.tolist()) + [bias_value])


def _export_threshold_and_conv_rows(writer, state_dict):
    exported = set()

    for name, param in state_dict.items():
        if name.endswith(".gatebank.w"):
            idx_key = name.replace(".w", ".idx")
            bias_key = name.replace(".w", ".theta")
            idx_tensor = state_dict.get(idx_key)
            if idx_tensor is None:
                continue
            bias_tensor = state_dict.get(bias_key)
            parameter_name = name.replace(".gatebank.w", ".effective_weight")
            _write_sparse_rows(writer, parameter_name, param, idx_tensor, bias_tensor)
            exported.add(name)
            continue

        if name.endswith(".weight"):
            idx_key = name[:-6] + "idx"
            idx_tensor = state_dict.get(idx_key)
            if idx_tensor is None:
                continue
            bias_key = name[:-6] + "bias"
            bias_tensor = state_dict.get(bias_key)
            parameter_name = name
            _write_sparse_rows(writer, parameter_name, param, idx_tensor, bias_tensor)
            exported.add(name)

    return exported

def export_effective_params_to_csv(model_weights_path, dataset_name, weights_dir="weights3/csvfiles", artifact_tag=None):
    """
    Loads a saved state_dict (.pth), builds effective_weight = weight * mask
    for masked layers, and writes model_effective_params_<dataset>.csv
    with up to 6 nonzero weights per neuron + bias.
    Args:
        model_weights_path (str): Path to the .pth file containing the model state_dict.
        dataset_name (str): Name of the dataset (used for naming the output CSV).
        weights_dir (str): Directory to save the output CSV file.
    Returns:
        str: Path to the generated CSV file.
    """
    os.makedirs(weights_dir, exist_ok=True)

    state_dict = _load_checkpoint_state_dict(model_weights_path)

    effective_state_dict = {}

    # build effective_state_dict: replace weight with weight * mask when mask exists
    for key, tensor in state_dict.items():
        if key.endswith("weight"):
            mask_key = key.replace("weight", "mask")
            if mask_key in state_dict:
                effective_weight = tensor * state_dict[mask_key]
                effective_state_dict[key.replace("weight", "effective_weight")] = effective_weight
            else:
                effective_state_dict[key] = tensor
        else:
            effective_state_dict[key] = tensor

    artifact_tag = _resolve_artifact_tag(dataset_name, model_weights_path=model_weights_path, artifact_tag=artifact_tag)
    csv_filename = os.path.join(weights_dir, f"model_effective_params_{artifact_tag}.csv")

    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'parameter', 'row_index', 'row_shape',
            'non_zero_indices', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6',
            'bias'
        ])

        exported_sparse_weights = _export_threshold_and_conv_rows(writer, state_dict)

        for name, param in effective_state_dict.items():
            if '.s' in name or '.mask' in name:
                continue

            if name in exported_sparse_weights:
                continue

            if not (name.endswith("weight") or name.endswith("effective_weight")):
                continue

            weight_np = param.detach().cpu().numpy()

            # infer bias key
            if name.endswith("effective_weight"):
                bias_key = name.replace("effective_weight", "bias")
            else:
                bias_key = name.replace("weight", "bias")

            bias_tensor = effective_state_dict.get(bias_key, None)
            bias_np = bias_tensor.detach().cpu().numpy() if bias_tensor is not None else None

            if weight_np.ndim == 1:
                row = weight_np
                nonzero_indices = np.nonzero(row)[0]
                nonzero_values = row[nonzero_indices]
                w_vals = [str(val) for val in nonzero_values]
                while len(w_vals) < 6:
                    w_vals.append("")
                bias_value = bias_np[0] if (bias_np is not None and bias_np.size > 0) else ""
                writer.writerow([
                    name,
                    0,
                    row.shape,
                    ','.join(map(str, nonzero_indices))
                ] + w_vals[:6] + [bias_value])
            else:
                for i, row in enumerate(weight_np):
                    if row.ndim > 1:
                        row = row.flatten()
                    nonzero_indices = np.nonzero(row)[0]
                    nonzero_values = row[nonzero_indices]
                    w_vals = [str(val) for val in nonzero_values]
                    while len(w_vals) < 6:
                        w_vals.append("")
                    bias_value = bias_np[i] if (bias_np is not None and i < bias_np.shape[0]) else ""
                    writer.writerow([
                        name,
                        i,
                        row.shape,
                        ','.join(map(str, nonzero_indices))
                    ] + w_vals[:6] + [bias_value])

    # optional: stats pass (kept, but not printed)
    df = pd.read_csv(csv_filename)
    grouped = df.groupby('parameter')
    for _, group in grouped:
        all_weight_values = []
        all_bias_values = []
        for _, row in group.iterrows():
            for col in ['w1', 'w2', 'w3', 'w4', 'w5', 'w6']:
                if pd.notna(row[col]) and row[col] != "":
                    try:
                        all_weight_values.append(float(row[col]))
                    except ValueError:
                        pass
            if pd.notna(row['bias']) and row['bias'] != "":
                try:
                    all_bias_values.append(float(row['bias']))
                except ValueError:
                    pass
        _ = np.array(all_weight_values)
        _ = np.array(all_bias_values)

    return csv_filename


# ──────────────────────────────────────────────────────────────────────────────
# 2) CSV post-processing: signs → bias, abs weights, reorder
# ──────────────────────────────────────────────────────────────────────────────

def process_input_csv(input_csv, output_csv):
    """
    Reads model_effective_params_<dataset>.csv, processes weights and bias:
      - Moves negative weights into bias
      - Converts weights to absolute values
      - Reorders weights in descending order of absolute value
    Writes processed data to model_effective_params_<dataset>_changed.csv
    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to the output CSV file.
    Returns:
        str: Path to the processed CSV file.
    """
    df = pd.read_csv(input_csv)

    df["parameter"] = df["parameter"].astype(str)
    df = df[~df["parameter"].str.contains(r"model\.2\.weight|model\.5\.weight", na=False)]

    if "non_zero_indices" not in df.columns:
        df["non_zero_indices"] = ""
    df["non_zero_indices"] = df["non_zero_indices"].fillna("").astype(str).astype("object")

    if "bias" not in df.columns:
        df["bias"] = 0.0
    df["bias"] = pd.to_numeric(df["bias"], errors="coerce").fillna(0.0)

    for i in range(1, 7):
        col = f"w{i}"
        if col not in df.columns:
            df[col] = float("nan")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # mark negative indices
    def add_negative_index(row):
        indices = [s.strip() for s in str(row["non_zero_indices"]).split(",") if s.strip()]
        negative_indices = []
        for i in range(1, 7):
            w = row.get(f"w{i}", 0.0)
            w = 0.0 if pd.isna(w) else float(w)
            if w < 0 and (i - 1) < len(indices):
                negative_indices.append(indices[i - 1])
        return ", ".join(negative_indices)

    df["negative index"] = df.apply(add_negative_index, axis=1).astype("object")

    # fold negatives into bias
    def update_bias(row):
        bias_val = float(row["bias"])
        for i in range(1, 7):
            w = row.get(f"w{i}", 0.0)
            w = 0.0 if pd.isna(w) else float(w)
            if w < 0:
                bias_val += -w
        return bias_val

    df["bias"] = df.apply(update_bias, axis=1)

    # absolute weights
    def convert_weights_to_abs(row):
        for i in range(1, 7):
            col = f"w{i}"
            w = row.get(col, None)
            if pd.isna(w):
                continue
            row[col] = abs(float(w))
        return row

    df = df.apply(convert_weights_to_abs, axis=1)

    df["non_zero_indices"] = df["non_zero_indices"].astype("object")
    df["negative index"] = df["negative index"].astype("object")

    def reorder_weights_indices_row(row):
        indices = [s.strip() for s in str(row["non_zero_indices"]).split(",") if s.strip()]
        originally_negative = {s.strip() for s in str(row["negative index"]).split(",") if s.strip()}

        pairs = []
        for i in range(1, 7):
            w = pd.to_numeric(row.get(f"w{i}", 0.0), errors="coerce")
            w = 0.0 if pd.isna(w) else float(w)
            idx = indices[i - 1] if i - 1 < len(indices) else ""
            was_neg = idx in originally_negative
            pairs.append((w, idx, was_neg))

        pairs.sort(key=lambda x: x[0], reverse=True)

        new_weights = [p[0] for p in pairs]
        new_nzi = ", ".join(p[1] for p in pairs if p[1])
        new_neg = ", ".join(p[1] for p in pairs if p[2] and p[1])

        out = {f"w{i+1}": new_weights[i] for i in range(6)}
        out["non_zero_indices_new"] = new_nzi
        out["negative index_new"] = new_neg
        return pd.Series(out)

    _out = df.apply(reorder_weights_indices_row, axis=1)

    for i in range(1, 7):
        df[f"w{i}"] = pd.to_numeric(_out[f"w{i}"], errors="coerce").fillna(0.0)

    df["non_zero_indices"] = _out["non_zero_indices_new"].astype("object")
    df["negative index"] = _out["negative index_new"].astype("object")

    df.to_csv(output_csv, index=False)
    return output_csv


# ──────────────────────────────────────────────────────────────────────────────
# 3) Truth-table generation from 6-weight neuron
# ──────────────────────────────────────────────────────────────────────────────

def generate_truth_tables(input_csv, output_truth_csv):
    """
    Reads model_effective_params_<dataset>_changed.csv and writes
    truth_tables_generated_<dataset>.csv with truth tables for each 6-weight neuron.
    Args:
        input_csv (str): Path to the processed CSV file.
        output_truth_csv (str): Path to the output truth table CSV file.
    Returns:
        str: Path to the generated truth table CSV file.
    """
    def generate_truth_table_dict_normal(w1, w2, w3, w4, w5, w6, bias):
        truth = {}
        for combo in itertools.product([0, 1], repeat=6):
            x1, x2, x3, x4, x5, x6 = combo
            f = w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + w6*x6
            truth["".join(str(b) for b in combo)] = 1 if f >= bias else 0
        return truth

    df = pd.read_csv(input_csv)
    functions_dict = {}

    for _, row in df.iterrows():
        try:
            w1 = float(row["w1"]); w2 = float(row["w2"]); w3 = float(row["w3"])
            w4 = float(row["w4"]); w5 = float(row["w5"]); w6 = float(row["w6"])
            bias = float(row["bias"])
        except Exception:
            continue

        try:
            neuron = int(row["row_index"])
        except Exception:
            continue

        param = str(row.get("parameter", ""))
        layer = _extract_sparse_layer_id(param)
        if layer is None:
            continue

        truth_table = generate_truth_table_dict_normal(w1, w2, w3, w4, w5, w6, bias)
        functions_dict[f"row{neuron}_{layer}"] = truth_table

    if not functions_dict:
        pd.DataFrame(columns=["Function", "neuron number"]).to_csv(output_truth_csv, index=False)
        return output_truth_csv

    first_truth = next(iter(functions_dict.values()))
    minterms = sorted(list(first_truth.keys()))

    rows = []
    for idx, (func, truth) in enumerate(functions_dict.items()):
        row_out = {"Function": func, "neuron number": f"neuron{idx}"}
        row_out.update({m: truth[m] for m in minterms})
        rows.append(row_out)

    pd.DataFrame(rows, columns=["Function", "neuron number"] + minterms).to_csv(output_truth_csv, index=False)
    return output_truth_csv


# ──────────────────────────────────────────────────────────────────────────────
# 4) Quine–McCluskey and minimization
# ──────────────────────────────────────────────────────────────────────────────

def _int_to_bits(n, num_vars):
    return tuple((n >> i) & 1 for i in reversed(range(num_vars)))

def _count_ones(bits):
    return sum(bits)

def _combine(a, b):
    diff = 0
    out  = []
    for x, y in zip(a, b):
        if x == y:
            out.append(x)
        else:
            diff += 1
            out.append(None)
    return tuple(out) if diff == 1 else None

def quine_mccluskey(truth, dont_care=None, var_names=None):
    """
    Minimizes a boolean function using the Quine–McCluskey algorithm.
    Args:
        truth (list of int): Truth table as a list of 0s and 1s.
        dont_care (list of int, optional): List of indices to treat as don't care conditions.
        var_names (list of str, optional): Variable names for the output expression.
    Returns:
        str: Minimized boolean expression in SOP form.
    """
    if dont_care is None:
        dont_care = []
    n = len(truth)
    m = n.bit_length() - 1
    if var_names is None:
        var_names = [f"x{i}" for i in range(m)]

    minterms  = {i for i, v in enumerate(truth) if v == 1}
    dc_terms  = set(dont_care)
    all_terms = minterms | dc_terms

    groups = {}
    for t in all_terms:
        bits = _int_to_bits(t, m)
        groups.setdefault(_count_ones(bits), []).append(bits)

    prime_implicants = set()
    while groups:
        new_groups = {}
        marked     = set()
        for k in sorted(groups):
            for a in groups[k]:
                for b in groups.get(k + 1, []):
                    c = _combine(a, b)
                    if c:
                        key = _count_ones([bit for bit in c if bit == 1])
                        new_groups.setdefault(key, []).append(c)
                        marked.add(a)
                        marked.add(b)
        for bucket in groups.values():
            for term in bucket:
                if term not in marked:
                    prime_implicants.add(term)
        groups = {k: list(set(v)) for k, v in new_groups.items()}

    chart = {}
    for pi in prime_implicants:
        covers = {
            t for t in minterms
            if all(b is None or b == tb for b, tb in zip(pi, _int_to_bits(t, m)))
        }
        if covers:
            chart[pi] = covers

    essential = []
    uncovered = set(minterms)
    for t in minterms:
        covering = [pi for pi, cov in chart.items() if t in cov]
        if len(covering) == 1 and covering[0] not in essential:
            essential.append(covering[0])
    for pi in essential:
        uncovered -= chart[pi]

    best = set()
    if uncovered:
        P = []
        for t in uncovered:
            pis = [pi for pi, cov in chart.items() if t in cov]
            P.append({frozenset([pi]) for pi in pis})

        def _mul(A, B): return {a | b for a in A for b in B}
        candidates = reduce(_mul, P)
        min_size   = min(len(sol) for sol in candidates)
        best       = next(sol for sol in candidates if len(sol) == min_size)

    final = set(essential) | set(best)

    def _term_to_str(term):
        parts = []
        for name, bit in zip(var_names, term):
            if bit == 1:
                parts.append(name)
            elif bit == 0:
                parts.append(f"~{name}")
        return "1" if not parts else "&".join(parts)

    return "|".join(_term_to_str(pi) for pi in final)


def minimize_truth_tables(dataset_name, csv_dir="weights3/csvfiles", truth_table_dir="weights3/truthtable", out_dir="weights3/out", artifact_tag=None):
    """
    Reads:
      - model_effective_params_<dataset_name>_changed.csv
      - truth_tables_generated_<dataset_name>.csv
    and writes:
      - minimized_expressions_<dataset_name>.txt
    """
    os.makedirs(out_dir, exist_ok=True)

    artifact_tag = _resolve_artifact_tag(dataset_name, artifact_tag=artifact_tag)

    proc_csv  = os.path.join(csv_dir, f"model_effective_params_{artifact_tag}_changed.csv")
    truth_csv = os.path.join(truth_table_dir, f"truth_tables_generated_{artifact_tag}.csv")
    output    = os.path.join(out_dir,    f"minimized_expressions_{artifact_tag}.txt")

    var_names = ['a', 'b', 'c', 'd', 'e', 'f']
    order     = {v: i for i, v in enumerate(var_names)}

    expr_ids = {}
    next_id  = 0

    df_proc = pd.read_csv(proc_csv)
    proc_map = {}
    for _, r in df_proc.iterrows():
        try:
            idx = int(r['row_index'])
        except Exception:
            continue
        param = str(r['parameter'])
        layer = _extract_sparse_layer_id(param)
        if layer is None:
            continue
        key = f"row{idx}_{layer}"
        proc_map[key] = {
            "non_zero": r.get('non_zero_indices', ""),
            "negative": r.get('negative index', "")
        }

    df_truth = pd.read_csv(truth_csv)
    if df_truth.empty:
        print(f"[minimize_truth_tables] No rows in {truth_csv}, skipping minimization.")
        return

    truth_cols = [c for c in df_truth.columns if c not in ['Function', 'neuron number']]

    with open(output, "w") as out:
        for _, r in df_truth.iterrows():
            func   = r['Function']
            neuron = r['neuron number']
            truth  = [int(r[c]) for c in truth_cols]

            if not any(truth):
                expr = "zero"
            else:
                expr = quine_mccluskey(truth, var_names=var_names)
                if expr == "1":
                    expr = "one"
                terms = expr.split("|") if expr else []
                def key_term(t):
                    if t in ("one", "zero", ""):
                        return [len(var_names)]
                    lits = [lit.lstrip("~") for lit in t.split("&")]
                    return [order.get(l, len(var_names)) for l in lits]
                expr = "|".join(sorted(terms, key=key_term)) if terms else "zero"

            fid = None
            if expr not in ("zero", "one"):
                if expr not in expr_ids:
                    expr_ids[expr] = next_id
                    next_id += 1
                fid = expr_ids[expr]

            info = proc_map.get(func, {"non_zero": "", "negative": ""})
            nz   = info["non_zero"]
            neg  = info["negative"]

            line = f"{func} / {neuron} / {expr}"
            if fid is not None:
                line += f" / (Func: {fid})"
            line += f" / non_zero_indices: {nz} / negative index: {neg}\n"
            out.write(line)

    print(f"[minimize_truth_tables] Wrote {len(expr_ids)} unique non-constant functions to {output}")


def decode_conv_indices(dataset_name, csv_dir="weights3/csvfiles", truth_table_dir="weights3/truthtable", out_dir="weights3/out", artifact_tag=None):
    """
    Decode SparseChannelLockedConv unfolded indices into channel-local 3x3 taps
    and add binary MaxPool2d rows as 2x2 OR gates.

    Writes:
      - decoded_indices_<dataset_name>.txt

    Only convolutional Sequential layers up to model.12 are decoded. MaxPool2d
    rows are emitted for the pooling layers through model.14.
    """
    os.makedirs(out_dir, exist_ok=True)

    artifact_tag = _resolve_artifact_tag(dataset_name, artifact_tag=artifact_tag)
    proc_csv = os.path.join(csv_dir, f"model_effective_params_{artifact_tag}_changed.csv")
    truth_csv = os.path.join(truth_table_dir, f"truth_tables_generated_{artifact_tag}.csv")
    minimized_txt = os.path.join(out_dir, f"minimized_expressions_{artifact_tag}.txt")
    output = os.path.join(out_dir, f"decoded_indices_{artifact_tag}.txt")

    conv_layers = {0, 3, 6, 9, 12}
    pool_layer_by_conv_layer = {0: 2, 3: 5, 6: 8, 9: 11, 12: 14}
    kernel_size = 3
    k2 = kernel_size * kernel_size

    if not os.path.exists(proc_csv):
        raise FileNotFoundError(f"Processed CSV not found: {proc_csv}")

    neuron_by_func = {}
    if os.path.exists(truth_csv):
        df_truth = pd.read_csv(truth_csv)
        for _, row in df_truth.iterrows():
            neuron_by_func[str(row.get("Function", ""))] = str(row.get("neuron number", ""))

    def parse_indices(value):
        if pd.isna(value):
            return []
        return [int(part.strip()) for part in str(value).split(",") if part.strip()]

    def fmt_list(values):
        return ", ".join(str(value) for value in values) if values else "none"

    df = pd.read_csv(proc_csv)
    decoded_by_func = {}
    for _, row in df.iterrows():
        layer = _extract_sparse_layer_id(str(row.get("parameter", "")))
        if layer not in conv_layers:
            continue

        try:
            row_index = int(row["row_index"])
        except Exception:
            continue

        indices = parse_indices(row.get("non_zero_indices", ""))
        negative_indices = set(parse_indices(row.get("negative index", "")))

        if not indices:
            continue

        channels = [idx // k2 for idx in indices]
        local_indices = [idx % k2 for idx in indices]
        negative_local_indices = [idx % k2 for idx in indices if idx in negative_indices]
        unique_channels = sorted(set(channels))

        decoded_by_func[f"row{row_index}_{layer}"] = {
            "layer": layer,
            "row_index": row_index,
            "channels": unique_channels,
            "local_indices": local_indices,
            "negative_local_indices": negative_local_indices,
        }

    pool_rows_by_layer = {}
    for conv_layer, pool_layer in pool_layer_by_conv_layer.items():
        conv_row_indices = sorted(
            info["row_index"]
            for info in decoded_by_func.values()
            if info["layer"] == conv_layer
        )
        pool_rows_by_layer[pool_layer] = [
            (
                f"row{channel}_{pool_layer} / OR gate / a|b|c|d "
                f"/ channel: {channel} / non_zero_indices: 0, 1, 2, 3\n"
            )
            for channel in conv_row_indices
        ]

    rows_written = 0
    pool_rows_written = 0

    with open(output, "w") as out:
        if os.path.exists(minimized_txt):
            with open(minimized_txt, "r") as infile:
                source_lines = infile.readlines()
        else:
            source_lines = []
            for func, info in decoded_by_func.items():
                neuron = neuron_by_func.get(func, "")
                prefix = f"{func}"
                if neuron:
                    prefix += f" / {neuron}"
                source_lines.append(prefix + "\n")

        previous_layer = None
        for raw_line in source_lines:
            line = raw_line.strip()
            if not line:
                continue

            func = line.split("/", 1)[0].strip()
            info = decoded_by_func.get(func)
            if info is None:
                continue

            layer = info["layer"]
            if previous_layer is not None and layer != previous_layer:
                pool_layer = pool_layer_by_conv_layer.get(previous_layer)
                for pool_line in pool_rows_by_layer.get(pool_layer, []):
                    out.write(pool_line)
                    pool_rows_written += 1
            previous_layer = layer

            prefix = line
            for marker in (" / non_zero_indices:", " / negative index:"):
                if marker in prefix:
                    prefix = prefix.split(marker, 1)[0].strip()

            out.write(
                f"{prefix} / channel: {fmt_list(info['channels'])} "
                f"/ non_zero_indices: {fmt_list(info['local_indices'])} "
                f"/ negative index: {fmt_list(info['negative_local_indices'])}\n"
            )
            rows_written += 1

        if previous_layer is not None:
            pool_layer = pool_layer_by_conv_layer.get(previous_layer)
            for pool_line in pool_rows_by_layer.get(pool_layer, []):
                out.write(pool_line)
                pool_rows_written += 1

    print(
        f"[decode_conv_indices] Wrote {rows_written} decoded conv rows and "
        f"{pool_rows_written} maxpool rows to {output}"
    )
    return output


# ──────────────────────────────────────────────────────────────────────────────
# 5) Single entrypoint
# ──────────────────────────────────────────────────────────────────────────────

def run_full_logic_pipeline(
    dataset_name,
    model_weights_path,
    csv_dir="weights3/csvfiles",
    truth_table_dir="weights3/truthtable",
    out_dir="weights3/out",
    artifact_tag=None,
):
    """
    One-shot pipeline:
      .pth → model_effective_params.csv
            → model_effective_params_changed.csv
            → truth_tables_generated.csv
            → minimized_expressions.txt
            → decoded_indices.txt
    """
    artifact_tag = _resolve_artifact_tag(dataset_name, model_weights_path=model_weights_path, artifact_tag=artifact_tag)

    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(truth_table_dir, exist_ok=True)

    export_effective_params_to_csv(
        model_weights_path,
        dataset_name,
        weights_dir=csv_dir,
        artifact_tag=artifact_tag,
    )

    original_csv    = os.path.join(csv_dir, f"model_effective_params_{artifact_tag}.csv")
    processed_csv   = os.path.join(csv_dir, f"model_effective_params_{artifact_tag}_changed.csv")
    truth_table_csv = os.path.join(truth_table_dir, f"truth_tables_generated_{artifact_tag}.csv")

    process_input_csv(original_csv, processed_csv)
    generate_truth_tables(processed_csv, truth_table_csv)
    minimize_truth_tables(
        dataset_name,
        csv_dir=csv_dir,
        truth_table_dir=truth_table_dir,
        out_dir=out_dir,
        artifact_tag=artifact_tag,
    )
    decode_conv_indices(
        dataset_name,
        csv_dir=csv_dir,
        truth_table_dir=truth_table_dir,
        out_dir=out_dir,
        artifact_tag=artifact_tag,
    )
