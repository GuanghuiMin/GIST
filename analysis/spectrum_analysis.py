#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import gc
from typing import Any, Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from less.data_selection.get_validation_dataset import get_dataset, get_dataloader
from less.data_selection.collect_grad_reps import obtain_gradients, prepare_batch


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']  # fallback helps ensure bold exists
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['legend.fontsize'] = 20  # you already set legend fontsize in code; keep this as a safe default


# -----------------------------
# Plot styling utilities
# -----------------------------
def stylize_axis(ax, labelsize=16, spine_lw=1.4, tick_lw=1.6, tick_len=7):
    for spine in ax.spines.values():
        spine.set_linewidth(spine_lw)
        spine.set_color("black")

    ax.tick_params(
        axis="both", which="major",
        direction="inout",
        length=tick_len,
        width=tick_lw,
        labelsize=labelsize
    )
    ax.tick_params(axis="both", which="minor", length=4, width=1.2)
    ax.grid(True, which="major", linestyle="--", alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", alpha=0.18)

    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")

    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    ax.title.set_fontweight("bold")


def make_epoch_styles(num_epochs: int):
    cmap = plt.get_cmap("tab10" if num_epochs <= 10 else "tab20")
    linestyles = ["-", "--", "-.", ":"]

    styles = {}
    for i in range(num_epochs):
        label = f"Epoch {i+1}"
        color = cmap(i % cmap.N)
        linestyle = linestyles[i % len(linestyles)]

        styles[label] = {
            "color": color,
            "linestyle": linestyle,
            "linewidth": 2.4,
        }
    return styles


# -----------------------------
# Chunked computations
# -----------------------------
def compute_gram_matrix_chunked_stream_to_device(
    G_cpu: torch.Tensor,
    device: torch.device,
    chunk_size: int = 10_000_000,
) -> torch.Tensor:
    """
    Compute K = G G^T with G stored on CPU.
    Only move a chunk of columns to GPU each iteration to avoid OOM.

    G_cpu: [N, D] float32 on CPU
    Returns: K_cpu [N, N] float32 on CPU
    """
    assert G_cpu.device.type == "cpu"
    N, D = G_cpu.shape

    K = torch.zeros((N, N), device=device, dtype=torch.float32)

    print(f"Computing Gram Matrix K=GG^T by streaming chunks (N={N}, D={D}, chunk_size={chunk_size}, device={device})...")
    for i in tqdm(range(0, D, chunk_size), desc="Gram Matrix Progress"):
        end = min(i + chunk_size, D)
        g_chunk = G_cpu[:, i:end].to(device, non_blocking=True)  # [N, chunk]
        K += g_chunk @ g_chunk.T
        del g_chunk
        if device.type == "cuda":
            torch.cuda.synchronize()

    return K.cpu()


def compute_projection_matrix_chunked_stream_to_device(
    G_cpu: torch.Tensor,
    W_cpu: torch.Tensor,
    device: torch.device,
    chunk_size: int = 5_000_000,
) -> torch.Tensor:
    """
    Compute P = G^T W with both stored on CPU, stream chunks to GPU.

    G_cpu: [N, D] float32 CPU
    W_cpu: [N, k] float32 CPU
    Returns: P_cpu [D, k] float32 CPU
    """
    assert G_cpu.device.type == "cpu" and W_cpu.device.type == "cpu"
    N, D = G_cpu.shape
    _, k = W_cpu.shape

    P_cpu = torch.empty((D, k), dtype=torch.float32, device="cpu")
    W = W_cpu.to(device, non_blocking=True)

    print(f"Computing Projection Matrix P=G^T W by streaming chunks (D={D}, k={k}, chunk_size={chunk_size}, device={device})...")
    for i in tqdm(range(0, D, chunk_size), desc="Projection Matrix Progress"):
        end = min(i + chunk_size, D)
        g_chunk_T = G_cpu[:, i:end].T.to(device, non_blocking=True)  # [chunk, N]
        p_chunk = g_chunk_T @ W  # [chunk, k]
        P_cpu[i:end, :] = p_chunk.detach().cpu()
        del g_chunk_T, p_chunk
        if device.type == "cuda":
            torch.cuda.synchronize()

    return P_cpu


# -----------------------------
# Model loading
# -----------------------------
def load_model(model_name_or_path: str, torch_dtype: Any = torch.bfloat16) -> Any:
    is_peft = os.path.exists(os.path.join(model_name_or_path, "adapter_config.json"))
    if is_peft:
        config = LoraConfig.from_pretrained(model_name_or_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, model_name_or_path, device_map="auto")
        base_for_tokenizer = config.base_model_name_or_path
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        base_for_tokenizer = model_name_or_path

    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True

    model.train()
    return model, base_for_tokenizer


@torch.no_grad()
def _resize_embeddings_if_needed(model, tokenizer):
    emb = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > emb:
        model.resize_token_embeddings(len(tokenizer))


def _model_primary_device(model) -> torch.device:
    return next(model.parameters()).device


# -----------------------------
# Core per-checkpoint analysis
# -----------------------------
def compute_spectrum_for_checkpoint(
    model_path: str,
    epoch_label: str,
    task: str,
    data_dir: str,
    max_length: int,
    max_samples: int,
    torch_dtype: Any,
    chunk_size: int,
) -> Dict[str, Any]:
    print("=" * 80)
    print(f"[{epoch_label}] Loading model from: {model_path}")
    model, base_for_tokenizer = load_model(model_path, torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(base_for_tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    _resize_embeddings_if_needed(model, tokenizer)

    model_device = _model_primary_device(model)
    compute_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(f"[{epoch_label}] Model primary device: {model_device} | Compute device for chunks: {compute_device}")

    print(f"[{epoch_label}] Loading dataset: {task}")
    dataset = get_dataset(task, data_dir=data_dir, tokenizer=tokenizer, max_length=max_length)
    dataloader = get_dataloader(dataset, tokenizer=tokenizer, batch_size=1)

    print(f"[{epoch_label}] Collecting gradients (max_samples={max_samples})...")
    raw_grads: List[torch.Tensor] = []
    count = 0

    for batch in tqdm(dataloader, desc=f"Collecting Gradients ({epoch_label})"):
        prepare_batch(batch, device=model_device)
        grad_vec = obtain_gradients(model, batch)
        raw_grads.append(grad_vec.detach().cpu().float())
        count += 1
        if max_samples is not None and count >= max_samples:
            break

    if len(raw_grads) == 0:
        raise RuntimeError(f"[{epoch_label}] No gradients collected.")

    G = torch.stack(raw_grads, dim=0).contiguous()  # [N, D] CPU float32
    N, D = G.shape
    print(f"[{epoch_label}] Gradient matrix: N={N} samples, D={D} params (stored on CPU)")

    K = compute_gram_matrix_chunked_stream_to_device(G, device=compute_device, chunk_size=chunk_size)

    print(f"[{epoch_label}] Eigendecomposition on K (CPU, N x N)...")
    eigenvalues, _ = torch.linalg.eigh(K)
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]

    S = torch.sqrt(torch.clamp(eigenvalues, min=0.0)).cpu()
    s2 = S ** 2
    explained = s2 / (s2.sum() + 1e-12)
    cumsum = torch.cumsum(explained, dim=0).cpu()

    def r_at(thr: float) -> int:
        hit = torch.where(cumsum >= thr)[0]
        return int(hit[0].item() + 1) if len(hit) else int(len(cumsum))

    r95 = r_at(0.95)
    r99 = r_at(0.99)
    print(f"[{epoch_label}] r95={r95}, r99={r99}")

    del model, tokenizer, dataset, dataloader, raw_grads
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"label": epoch_label, "S": S, "cumsum": cumsum, "r95": r95, "r99": r99, "N": N, "D": D}


# -----------------------------
# Plotting
# -----------------------------
def plot_singular_multi(results: List[Dict[str, Any]], out_path: str, max_plot_rank: int, title: str):
    fig, ax = plt.subplots(figsize=(8.6, 6.3))
    styles = make_epoch_styles(len(results))

    for res in results:
        label = res["label"]
        S = res["S"].numpy()
        if max_plot_rank and max_plot_rank > 0:
            S = S[:max_plot_rank]
        x = np.arange(1, len(S) + 1)

        st = styles[label]
        ax.plot(
            x, S, label=label,
            color=st["color"], linestyle=st["linestyle"], linewidth=st["linewidth"], alpha=0.95,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Component index", fontsize=24, fontweight="bold")
    ax.set_ylabel("Singular value (log scale)", fontsize=24, fontweight="bold")
    if max_plot_rank and max_plot_rank > 0:
        ax.set_xlim(1, max_plot_rank)

    stylize_axis(ax, labelsize=16)

    leg = ax.legend(
        loc="upper right", bbox_to_anchor=(1.02, 1.02),
        fontsize=20, frameon=True, framealpha=1,
        facecolor="white", edgecolor="black",
        prop={"weight": "bold"},  # force bold legend text
    )
    leg.get_frame().set_linewidth(1.2)

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")


def plot_cumulative_multi(results: List[Dict[str, Any]], out_path: str, max_plot_rank: int, title: str, annotate_r95: bool = True):
    fig, ax = plt.subplots(figsize=(8.6, 6.3))
    styles = make_epoch_styles(len(results))

    for res in results:
        label = res["label"]
        c = res["cumsum"].numpy()
        if max_plot_rank and max_plot_rank > 0:
            c = c[:max_plot_rank]
        x = np.arange(1, len(c) + 1)

        st = styles[label]
        ax.plot(
            x, c, label=label,
            color=st["color"], linestyle=st["linestyle"], linewidth=st["linewidth"],
            alpha=0.95,
        )

    ax.axhline(0.95, linestyle="--", linewidth=2.0, alpha=0.9, color="black", label="95% variance")

    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Component index", fontsize=24, fontweight="bold")
    ax.set_ylabel("Cumulative explained variance", fontsize=24, fontweight="bold")
    if max_plot_rank and max_plot_rank > 0:
        ax.set_xlim(1, max_plot_rank)

    stylize_axis(ax, labelsize=16)

    leg = ax.legend(
        loc="lower right", bbox_to_anchor=(1.02, -0.01),
        fontsize=20, frameon=True, framealpha=1,
        facecolor="white", edgecolor="black",
        prop={"weight": "bold"},  # force bold legend text
    )
    leg.get_frame().set_linewidth(1.2)

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Multi-checkpoint val-gradient spectrum plotter (fixed)")

    p.add_argument("--task", type=str, required=True)
    p.add_argument("--data_dir", type=str, default="../data")
    p.add_argument("--model_paths", type=str, nargs="+", required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--chunk_size", type=int, default=10_000_000)
    p.add_argument("--max_plot_rank", type=int, default=200)

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dtype = torch.float32 if args.torch_dtype == "float32" else torch.bfloat16

    results: List[Dict[str, Any]] = []
    for i, mp in enumerate(args.model_paths, start=1):
        epoch_label = f"Epoch {i}"
        res = compute_spectrum_for_checkpoint(
            model_path=mp,
            epoch_label=epoch_label,
            task=args.task,
            data_dir=args.data_dir,
            max_length=args.max_length,
            max_samples=args.max_samples,
            torch_dtype=dtype,
            chunk_size=args.chunk_size,
        )
        results.append(res)

    singular_out = os.path.join(args.output_dir, "singular_values_multi.pdf")
    cum_out = os.path.join(args.output_dir, "explained_variance_multi.pdf")

    plot_singular_multi(results, singular_out, args.max_plot_rank, "")
    plot_cumulative_multi(results, cum_out, args.max_plot_rank, "", annotate_r95=True)

    print("\nDone.")


if __name__ == "__main__":
    main()