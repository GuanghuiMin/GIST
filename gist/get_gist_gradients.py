import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Any
import gc
import time

from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from gist.data_selection.get_validation_dataset import get_dataset, get_dataloader
from gist.data_selection.collect_grad_reps import obtain_gradients, prepare_batch
from gist.data_selection.get_training_dataset import get_training_dataset

class PerformanceMonitor:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.events = []
        self.total_flops = 0
        self.start_time = time.time()
        
        with open(self.log_file_path, 'w') as f:
            f.write("=== GIST End-to-End Performance Log ===\n")
            f.write(f"Start Time: {time.ctime()}\n\n")

    def log_event(self, phase_name, duration, flops=0, note=""):
        self.events.append({
            "phase": phase_name,
            "duration": duration,
            "flops": flops,
            "note": note
        })
        self.total_flops += flops
        
        msg = f"[{phase_name}] Duration: {duration:.2f}s"
        if flops > 0:
            msg += f" | FLOPs: {flops:.2e} ({flops/1e12:.4f} TFLOPs)"
            if duration > 0:
                tflops_per_sec = (flops / 1e12) / duration
                msg += f" | Speed: {tflops_per_sec:.4f} TFLOPS"
        if note:
            msg += f" | Note: {note}"
        
        print(f"\n[LOG] {msg}")
        with open(self.log_file_path, 'a') as f:
            f.write(msg + "\n")

    def summary(self):
        total_time = time.time() - self.start_time
        with open(self.log_file_path, 'a') as f:
            f.write("\n=== Summary ===\n")
            f.write(f"Total Wall Time: {total_time:.2f}s ({total_time/60:.2f}m)\n")
            f.write(f"Total FLOPs: {self.total_flops:.2e} ({self.total_flops/1e15:.4f} PFLOPs)\n")
            if total_time > 0:
                avg_speed = (self.total_flops / 1e12) / total_time
                f.write(f"Average Effective Speed: {avg_speed:.4f} TFLOPS\n")
            f.write("===================\n")

def load_model(model_name_or_path: str, torch_dtype: Any = torch.bfloat16) -> Any:
    is_peft = os.path.exists(os.path.join(model_name_or_path, "adapter_config.json"))
    if is_peft:
        config = LoraConfig.from_pretrained(model_name_or_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, torch_dtype=torch_dtype, device_map="auto")
        model = PeftModel.from_pretrained(base_model, model_name_or_path, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch_dtype, device_map="auto")

    for name, param in model.named_parameters():
        if 'lora' in name or 'Lora' in name:
            param.requires_grad = True
    return model

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-End GIST (VRAM Optimized + Logging)')
    parser.add_argument('--task', type=str, required=True, help='Validation Task Name')
    parser.add_argument('--train_files', type=str, nargs='+', required=True, help='Path(s) to Training Data JSONL')
    parser.add_argument('--train_file_names', type=str, nargs='+', help='Custom names for output files')
    
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--target_dim', type=int, default=150, help='Rank k')
    
    parser.add_argument('--proj_dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--use_gpu_proj', action='store_true', help='If set, try to move P to GPU.')
    
    parser.add_argument('--max_val_samples', type=int, default=None)
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--chunk_size', type=int, default=10_000_000)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    monitor = PerformanceMonitor(os.path.join(args.output_dir, "gist_performance.log"))

    if args.train_file_names is None:
        args.train_file_names = [os.path.splitext(os.path.basename(f))[0] for f in args.train_files]
    assert len(args.train_files) == len(args.train_file_names)

    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    proj_dtype = dtype_map[args.proj_dtype]

    t0 = time.time()
    print(f"Loading Model: {args.model_path}")
    model_dtype = torch.float16 if args.torch_dtype == "float16" else torch.bfloat16
    model = load_model(args.model_path, model_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None: tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    monitor.log_event("Model Loading", time.time() - t0)

    # ================= PHASE 1: Validation Set & SVD =================
    print(f"\n>>> [Phase 1] Processing Validation Task: {args.task}")
    t_phase1_start = time.time()
    
    val_dataset = get_dataset(args.task, data_dir=args.data_dir, tokenizer=tokenizer, max_length=args.max_length)
    val_dataloader = get_dataloader(val_dataset, tokenizer=tokenizer, batch_size=1)
    
    val_grads = []
    count = 0
    t0 = time.time()
    for batch in tqdm(val_dataloader, desc="Val Grads"):
        prepare_batch(batch, device=device)
        grad_vec = obtain_gradients(model, batch) # (D,) usually on GPU
        val_grads.append(grad_vec.cpu().float())  
        count += 1
        if args.max_val_samples is not None and count >= args.max_val_samples:
            break
    
    if not val_grads: return
    G_val = torch.stack(val_grads)
    N, D = G_val.shape
    monitor.log_event("Collect Val Grads", time.time() - t0, note=f"N={N}, D={D}")
    
    torch.cuda.empty_cache()

    # SVD: Gram Matrix
    print("Computing Gram Matrix...")
    t0 = time.time()
    K = torch.zeros((N, N), dtype=torch.float32)
    
    # FLOPs: N * D * N * 2 (Symmetric)
    gram_flops = 2 * N * N * D 

    for i in tqdm(range(0, D, args.chunk_size), desc="Gram Chunk"):
        end = min(i + args.chunk_size, D)
        g_chunk = G_val[:, i:end].to(device)
        K += torch.matmul(g_chunk, g_chunk.T).cpu()
        del g_chunk 
    
    monitor.log_event("Gram Matrix Calc", time.time() - t0, flops=gram_flops)
    torch.cuda.empty_cache()

    # Eigendecomposition
    t0 = time.time()
    eigenvalues, eigenvectors = torch.linalg.eigh(K.to(device))
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    S = torch.sqrt(torch.clamp(eigenvalues, min=0)).cpu()
    k = min(args.target_dim, len(S))
    monitor.log_event("Eigendecomposition", time.time() - t0, note=f"Rank k={k}")

    # Validation Gradient
    val_projected = eigenvectors[:, :k].cpu() * S[:k].unsqueeze(0)
    val_save_path = os.path.join(args.output_dir, "val_gist_grads.pt")
    torch.save(val_projected, val_save_path)

    # ================= PHASE 1.5: Build Projection Matrix =================
    print(f"Building Projection Matrix P ({args.proj_dtype}) on CPU...")
    t0 = time.time()
    
    S_k_inv = 1.0 / (S[:k] + 1e-6)
    U_k = eigenvectors[:, :k].cpu()
    W = U_k * S_k_inv.unsqueeze(0)
    W = W.to(device).float()
    
    P = torch.empty((D, k), dtype=proj_dtype, device='cpu')
    
    # FLOPs: D * N * k * 2
    build_p_flops = 2 * D * N * k

    for i in tqdm(range(0, D, args.chunk_size), desc="Building P"):
        end = min(i + args.chunk_size, D)
        g_chunk_T = G_val[:, i:end].to(device).float().T 
        p_chunk = torch.matmul(g_chunk_T, W)
        P[i:end, :] = p_chunk.to(proj_dtype).cpu()
        del g_chunk_T, p_chunk
    
    monitor.log_event("Build Projection P", time.time() - t0, flops=build_p_flops)
    
    del G_val, K, eigenvectors, val_grads, W
    torch.cuda.empty_cache()
    gc.collect()

    using_gpu = False
    if args.use_gpu_proj:
        try:
            t_move = time.time()
            P = P.to(device)
            using_gpu = True
            monitor.log_event("Move P to GPU", time.time() - t_move, note="Success")
        except RuntimeError:
            print("FAILED to move P to GPU. Fallback to CPU.")
            P = P.cpu()
            torch.cuda.empty_cache()

    # ================= PHASE 2: Processing Training Files =================
    print(f"\n>>> [Phase 2] Processing {len(args.train_files)} Training Files")
    
    total_train_samples = 0
    t_phase2_start = time.time()

    for train_file, name in zip(args.train_files, args.train_file_names):
        print(f"\nProcessing: {name}")
        t_file_start = time.time()
        
        train_dataset = get_training_dataset([train_file], tokenizer, args.max_length, sample_percentage=1.0)
        cols_to_keep = {"input_ids", "attention_mask", "labels"}
        train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c not in cols_to_keep])
        train_dataloader = get_dataloader(train_dataset, tokenizer=tokenizer, batch_size=1)
        
        file_grads = []
        count = 0
        total_grad_norm = 0.0
        for batch in tqdm(train_dataloader, desc=f"Proj {name}"):
            prepare_batch(batch, device=device)
            
            grad_vec = obtain_gradients(model, batch)
            
            current_norm = grad_vec.norm().item()
            total_grad_norm += current_norm

            if using_gpu:
                grad_vec = grad_vec.to(proj_dtype)
                low_dim = torch.matmul(grad_vec, P)
            else:
                grad_vec = grad_vec.cpu().to(proj_dtype)
                low_dim = torch.matmul(grad_vec, P)
            
            file_grads.append(low_dim.cpu().float())
            del grad_vec
            count += 1
            if args.max_train_samples and count >= args.max_train_samples: break
        
        torch.cuda.empty_cache()

        if file_grads:
            stacked_grads = torch.stack(file_grads)
            save_path = os.path.join(args.output_dir, f"{name}_gist_grads.pt")
            torch.save(stacked_grads, save_path)
            
            avg_norm = total_grad_norm / count if count > 0 else 0.0

            # FLOPs: Samples * D * k * 2
            file_flops = 2 * count * D * k
            monitor.log_event(f"Process {name}", time.time() - t_file_start, 
                              flops=file_flops, 
                              note=f"Samples={count}, AvgGradNorm={avg_norm:.4f}")
            total_train_samples += count
        
        del file_grads, train_dataset, train_dataloader
        gc.collect()

    monitor.summary()
    print(f"\nLog saved to {monitor.log_file_path}")
    print("\nDone! All files processed.")

if __name__ == "__main__":
    main()