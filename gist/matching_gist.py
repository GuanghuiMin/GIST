import argparse
import os
import torch
from tqdm import tqdm
import torch.nn.functional as F

TASK_CONFIG = {
    "tydiqa": 9,
    "mmlu": 57,
    "bbh": 23,
}

def parse_args():
    parser = argparse.ArgumentParser(description='Cosine Similarity Matching (Normalized Gradients)')
    parser.add_argument('--output_path', type=str, default="selected_data_cosine", help='Output dir')
    parser.add_argument('--gist_path_template', type=str, required=True)
    parser.add_argument('--target_task', type=str, default="mmlu", help='Target task name')
    parser.add_argument('--train_file_names', type=str, nargs='+', required=True, help='Train datasets')
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True, help='Checkpoints list')
    parser.add_argument('--checkpoint_weights', type=float, nargs='+', required=True, help='Weights list')
    args = parser.parse_args()
    return args

def load_tensor(path, device):
    if not os.path.exists(path): return None
    data = torch.load(path, map_location='cpu')
    if not torch.is_tensor(data): data = torch.tensor(data)
    return data.to(device).float()

def find_file(base_dir, prefix):
    p1 = os.path.join(base_dir, "all_orig.pt")
    if os.path.exists(p1): return p1
    p2 = os.path.join(base_dir, f"{prefix}_gist_grads.pt")
    if os.path.exists(p2): return p2
    return None

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_subtasks = TASK_CONFIG.get(args.target_task, 1)
    print(f"Running COSINE SIMILARITY for {args.target_task}")
    print(f"Subtasks: {n_subtasks} | Device: {device}")
    print("Strategy: Gradients -> Normalize -> Dot Product (Cosine) -> Max")

    raw_weights = torch.tensor(args.checkpoint_weights, dtype=torch.float32)
    weights = (raw_weights / raw_weights.sum()).tolist()

    # 1. Pre-calculate Validation Vectors
    print("\n>>> Pre-calculating Per-Task Vectors (Normalized)...")
    ckpt_resources = {} 
    
    for ckpt in tqdm(args.checkpoints, desc="Preparing Checkpoints"):
        try: base_path = args.gist_path_template.format(ckpt=ckpt)
        except: base_path = args.gist_path_template.replace("{ckpt}", str(ckpt))
            
        val_file_path = find_file(base_path, "val")
        if not val_file_path: continue

        val_info = load_tensor(val_file_path, device)

        
        # === Task-Aware Reshape ===
        n_val_total = val_info.shape[0]

        # Try grouping by subtasks
        if n_subtasks > 0 and n_val_total % n_subtasks == 0:
            # 1. Take mean of RAW gradients
            val_per_task = val_info.reshape(n_subtasks, -1, val_info.shape[-1]).mean(dim=1)
        else:
            print(f"Warning: Val size {n_val_total} is not divisible by subtasks {n_subtasks}. Treating each sample as a task.")
            val_per_task = val_info

        # 2. Normalize the resulting task vector (Essential for Cosine Similarity)
        val_vector = F.normalize(val_per_task, p=2, dim=1)
        
        ckpt_resources[ckpt] = val_vector
        del val_info, val_per_task
        torch.cuda.empty_cache()

    output_dir = os.path.join(args.output_path, args.target_task)
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    print(f"\n>>> Calculating Cosine Scores...")
    
    for train_file_name in args.train_file_names:
        print(f"Processing {train_file_name}...")
        
        accumulated_scores = None # Shape: [N_train, N_Tasks]
        
        for i, ckpt in enumerate(args.checkpoints):
            if ckpt not in ckpt_resources: continue
            try: base_path = args.gist_path_template.format(ckpt=ckpt)
            except: base_path = args.gist_path_template.replace("{ckpt}", str(ckpt))
            train_file_path = find_file(base_path, train_file_name)
            if not train_file_path: continue
            
            # Load Training Gradients
            training_info = load_tensor(train_file_path, device)
            
            training_info = F.normalize(training_info, p=2, dim=1)
            
            val_matrix = ckpt_resources[ckpt]
            
            # Cosine Similarity Calculation
            # Since inputs are normalized: A . B = |A||B|cos(theta) = 1*1*cos(theta)
            # [N, D] @ [N_Tasks, D].T -> [N, N_Tasks]
            current_score_matrix = torch.matmul(training_info, val_matrix.T)
            
            w = weights[i]
            if accumulated_scores is None:
                accumulated_scores = current_score_matrix * w
            else:
                # Handle potential shape mismatch (drop last batch issues)
                min_len = min(accumulated_scores.shape[0], current_score_matrix.shape[0])
                if accumulated_scores.shape[0] != current_score_matrix.shape[0]:
                    accumulated_scores = accumulated_scores[:min_len]
                    current_score_matrix = current_score_matrix[:min_len]
                
                accumulated_scores += current_score_matrix * w
            
            del training_info, current_score_matrix
        
        if accumulated_scores is None: continue

        # === Max Aggregation ===
        final_scores = accumulated_scores.max(dim=1)[0]

        output_file = os.path.join(output_dir, f"{train_file_name}_influence_score.pt")
        torch.save(final_scores.cpu(), output_file)
        print(f"  -> Saved {output_file} (Max: {final_scores.max():.4f})")
        
        del accumulated_scores, final_scores
        torch.cuda.empty_cache()

    print("Done!")

if __name__ == "__main__":
    main()