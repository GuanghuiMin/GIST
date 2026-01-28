# import argparse
# import os

# import torch

# argparser = argparse.ArgumentParser(
#     description='Script for selecting the data for training')
# argparser.add_argument('--gradient_path', type=str, default="{} ckpt{}",
#                        help='The path to the gradient file')
# argparser.add_argument('--train_file_names', type=str, nargs='+',
#                        help='The name of the training file')
# argparser.add_argument('--ckpts', type=int, nargs='+',
#                        help="Checkpoint numbers.")
# argparser.add_argument('--checkpoint_weights', type=float, nargs='+',
#                        help="checkpoint weights")
# argparser.add_argument('--target_task_names', type=str,
#                        nargs='+', help="The name of the target tasks")
# argparser.add_argument('--validation_gradient_path', type=str,
#                        default="{} ckpt{}", help='The path to the validation gradient file')
# argparser.add_argument('--output_path', type=str, default="selected_data",
#                        help='The path to the output')


# args = argparser.parse_args()

# N_SUBTASKS = {"mmlu": 57, "bbh": 27, "tydiqa": 9}

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def calculate_influence_score(training_info: torch.Tensor, validation_info: torch.Tensor):
#     """Calculate the influence score.

#     Args:
#         training_info (torch.Tensor): training info (gradients/representations) stored in a tensor of shape N x N_DIM
#         validation_info (torch.Tensor): validation info (gradients/representations) stored in a tensor of shape N_VALID x N_DIM
#     """
#     # N x N_VALID
#     influence_scores = torch.matmul(
#         training_info, validation_info.transpose(0, 1))
#     return influence_scores


# # renormalize the checkpoint weights
# if sum(args.checkpoint_weights) != 1:
#     s = sum(args.checkpoint_weights)
#     args.checkpoint_weights = [i/s for i in args.checkpoint_weights]

# # calculate the influence score for each validation task
# for target_task_name in args.target_task_names:
#     for train_file_name in args.train_file_names:
#         influence_score = 0
#         for i, ckpt in enumerate(args.ckpts):
#             # validation_path = args.validation_gradient_path.format(
#             # target_task_name, ckpt)
#             validation_path = args.validation_gradient_path.format(
#                 ckpt, target_task_name)
#             if os.path.isdir(validation_path):
#                 validation_path = os.path.join(validation_path, "all_orig.pt")
#             validation_info = torch.load(validation_path)

#             if not torch.is_tensor(validation_info):
#                 validation_info = torch.tensor(validation_info)
#             validation_info = validation_info.to(device).float()
#             # gradient_path = args.gradient_path.format(train_file_name, ckpt)
#             gradient_path = args.gradient_path.format(ckpt, train_file_name)
#             if os.path.isdir(gradient_path):
#                 gradient_path = os.path.join(gradient_path, "all_orig.pt")
#             training_info = torch.load(gradient_path)

#             if not torch.is_tensor(training_info):
#                 training_info = torch.tensor(training_info)
#             training_info = training_info.to(device).float()

#             influence_score += args.checkpoint_weights[i] * \
#                 calculate_influence_score(
#                     training_info=training_info, validation_info=validation_info)
#         influence_score = influence_score.reshape(
#             influence_score.shape[0], N_SUBTASKS[target_task_name], -1).mean(-1).max(-1)[0]
#         output_dir = os.path.join(args.output_path, target_task_name)
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#         output_file = os.path.join(
#             args.output_path, target_task_name, f"{train_file_name}_influence_score.pt")
#         torch.save(influence_score, output_file)
#         print("Saved influence score to {}".format(output_file))


# import argparse
# import os
# import torch
# import sys

# argparser = argparse.ArgumentParser(
#     description='Script for selecting the data for training')
# argparser.add_argument('--gradient_path', type=str, default="{} ckpt{}",
#                         help='The path to the gradient file')
# argparser.add_argument('--train_file_names', type=str, nargs='+',
#                         help='The name of the training file')
# argparser.add_argument('--ckpts', type=int, nargs='+',
#                         help="Checkpoint numbers.")
# argparser.add_argument('--checkpoint_weights', type=float, nargs='+',
#                         help="checkpoint weights")
# argparser.add_argument('--target_task_names', type=str,
#                         nargs='+', help="The name of the target tasks")
# argparser.add_argument('--validation_gradient_path', type=str,
#                         default="{} ckpt{}", help='The path to the validation gradient file')
# argparser.add_argument('--output_path', type=str, default="selected_data",
#                         help='The path to the output')


# args = argparser.parse_args()

# N_SUBTASKS = {"mmlu": 57, "bbh": 27, "tydiqa": 9}

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# def calculate_influence_score(training_info: torch.Tensor, validation_info: torch.Tensor):
#     """Calculate the influence score.

#     Args:
#         training_info (torch.Tensor): training info (gradients/representations) stored in a tensor of shape N x N_DIM
#         validation_info (torch.Tensor): validation info (gradients/representations) stored in a tensor of shape N_VALID x N_DIM
#     """
#     # N x N_VALID
#     influence_scores = torch.matmul(
#         training_info, validation_info.transpose(0, 1))
#     return influence_scores


# # renormalize the checkpoint weights
# if sum(args.checkpoint_weights) != 1:
#     s = sum(args.checkpoint_weights)
#     args.checkpoint_weights = [i/s for i in args.checkpoint_weights]

# print(f"\n>>> 开始检查维度一致性...")

# # calculate the influence score for each validation task
# for target_task_name in args.target_task_names:
#     for train_file_name in args.train_file_names:
#         influence_score = None  # 改为 None 以便区分初始化状态
        
#         print(f"\n=== Processing Task: {target_task_name} | Train File: {train_file_name} ===")

#         for i, ckpt in enumerate(args.ckpts):
#             # 1. Load Validation Info
#             validation_path = args.validation_gradient_path.format(
#                 ckpt, target_task_name)
#             if os.path.isdir(validation_path):
#                 validation_path = os.path.join(validation_path, "all_orig.pt")
            
#             # 简单检查 Validation 文件是否存在
#             if not os.path.exists(validation_path):
#                 print(f"[Error] Validation file not found: {validation_path}")
#                 sys.exit(1)
                
#             validation_info = torch.load(validation_path, map_location=device)
#             if not torch.is_tensor(validation_info):
#                 validation_info = torch.tensor(validation_info)
#             validation_info = validation_info.float()

#             # 2. Load Training Info (Gradients)
#             gradient_path = args.gradient_path.format(ckpt, train_file_name)
#             if os.path.isdir(gradient_path):
#                 gradient_path = os.path.join(gradient_path, "all_orig.pt")
            
#             # 简单检查 Training 文件是否存在
#             if not os.path.exists(gradient_path):
#                 print(f"[Error] Training gradient file not found: {gradient_path}")
#                 sys.exit(1)

#             training_info = torch.load(gradient_path, map_location=device)
#             if not torch.is_tensor(training_info):
#                 training_info = torch.tensor(training_info)
#             training_info = training_info.float()

#             # 3. 打印当前形状 (DEBUG核心)
#             print(f"  [Ckpt {ckpt}] Training Shape: {training_info.shape}, Validation Shape: {validation_info.shape}")

#             # 4. 计算当前项
#             current_term = args.checkpoint_weights[i] * \
#                 calculate_influence_score(
#                     training_info=training_info, validation_info=validation_info)
            
#             # 5. 累加并检查维度
#             if influence_score is None:
#                 # 第一次循环，直接赋值
#                 influence_score = current_term
#                 first_shape = influence_score.shape
#             else:
#                 # 后续循环，检查形状
#                 if influence_score.shape != current_term.shape:
#                     print("\n" + "!"*60)
#                     print(f"❌ ERROR: DIMENSION MISMATCH DETECTED!")
#                     print(f"Task Name   : {target_task_name}")
#                     print(f"Train File  : {train_file_name}")
#                     print(f"Current Ckpt: {ckpt}")
#                     print(f"-"*30)
#                     print(f"Expected Shape (based on previous ckpts): {first_shape}")
#                     print(f"Actual Shape (current ckpt {ckpt})      : {current_term.shape}")
#                     print(f"-"*30)
#                     print(f"Diagnosis: The gradient file for checkpoint {ckpt} has a different number of training samples ({training_info.shape[0]}) compared to the previous checkpoints.")
#                     print("!"*60 + "\n")
#                     raise RuntimeError(f"Dimension mismatch at Checkpoint {ckpt}")
                
#                 influence_score += current_term

#         # Reshape and Save logic
#         if influence_score is not None:
#             influence_score = influence_score.reshape(
#                 influence_score.shape[0], N_SUBTASKS[target_task_name], -1).mean(-1).max(-1)[0]
#             output_dir = os.path.join(args.output_path, target_task_name)
#             if not os.path.exists(output_dir):
#                 os.makedirs(output_dir)
#             output_file = os.path.join(
#                 args.output_path, target_task_name, f"{train_file_name}_influence_score.pt")
#             torch.save(influence_score, output_file)
#             print(f"✅ Saved influence score to {output_file}")

import argparse
import os
import torch
import sys

argparser = argparse.ArgumentParser(
    description='Script for selecting the data for training')
argparser.add_argument('--gradient_path', type=str, default="{} ckpt{}",
                        help='The path to the gradient file')
argparser.add_argument('--train_file_names', type=str, nargs='+',
                        help='The name of the training file')
argparser.add_argument('--ckpts', type=int, nargs='+',
                        help="Checkpoint numbers.")
argparser.add_argument('--checkpoint_weights', type=float, nargs='+',
                        help="checkpoint weights")
argparser.add_argument('--target_task_names', type=str,
                        nargs='+', help="The name of the target tasks")
argparser.add_argument('--validation_gradient_path', type=str,
                        default="{} ckpt{}", help='The path to the validation gradient file')
argparser.add_argument('--output_path', type=str, default="selected_data",
                        help='The path to the output')


args = argparser.parse_args()

N_SUBTASKS = {"mmlu": 57, "bbh": 27, "tydiqa": 9}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def calculate_influence_score(training_info: torch.Tensor, validation_info: torch.Tensor):
    """Calculate the influence score."""
    # N x N_VALID
    influence_scores = torch.matmul(
        training_info, validation_info.transpose(0, 1))
    return influence_scores

# renormalize the checkpoint weights
if sum(args.checkpoint_weights) != 1:
    s = sum(args.checkpoint_weights)
    args.checkpoint_weights = [i/s for i in args.checkpoint_weights]

print(f"\n>>> 开始计算分数 (包含自动对齐逻辑)...")

for target_task_name in args.target_task_names:
    for train_file_name in args.train_file_names:
        influence_score = None 
        
        print(f"\n=== Processing Task: {target_task_name} | Train File: {train_file_name} ===")

        for i, ckpt in enumerate(args.ckpts):
            # 1. Load Validation Info
            validation_path = args.validation_gradient_path.format(
                ckpt, target_task_name)
            if os.path.isdir(validation_path):
                validation_path = os.path.join(validation_path, "all_orig.pt")
            
            validation_info = torch.load(validation_path, map_location=device)
            if not torch.is_tensor(validation_info):
                validation_info = torch.tensor(validation_info)
            validation_info = validation_info.float()

            # 2. Load Training Info
            gradient_path = args.gradient_path.format(ckpt, train_file_name)
            if os.path.isdir(gradient_path):
                gradient_path = os.path.join(gradient_path, "all_orig.pt")
            
            training_info = torch.load(gradient_path, map_location=device)
            if not torch.is_tensor(training_info):
                training_info = torch.tensor(training_info)
            training_info = training_info.float()

            # 3. 计算当前项
            current_term = args.checkpoint_weights[i] * \
                calculate_influence_score(
                    training_info=training_info, validation_info=validation_info)
            
            # === 核心修改：自动对齐逻辑 ===
            if influence_score is None:
                influence_score = current_term
            else:
                # 获取两个张量的最小长度
                min_len = min(influence_score.shape[0], current_term.shape[0])
                
                # 如果当前项比之前的长，截断当前项
                if current_term.shape[0] > min_len:
                    print(f"⚠️ [Warning] Ckpt {ckpt}: Truncating current term from {current_term.shape[0]} to {min_len}")
                    current_term = current_term[:min_len]
                
                # 如果之前的比当前的长，截断之前的累加结果
                if influence_score.shape[0] > min_len:
                    print(f"⚠️ [Warning] Ckpt {ckpt}: Truncating accumulated score from {influence_score.shape[0]} to {min_len}")
                    influence_score = influence_score[:min_len]
                
                # 现在长度一致了，可以相加
                influence_score += current_term
            # ============================

        # Save logic
        if influence_score is not None:
            influence_score = influence_score.reshape(
                influence_score.shape[0], N_SUBTASKS[target_task_name], -1).mean(-1).max(-1)[0]
            output_dir = os.path.join(args.output_path, target_task_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_file = os.path.join(
                args.output_path, target_task_name, f"{train_file_name}_influence_score.pt")
            torch.save(influence_score, output_file)
            print(f"✅ Saved influence score to {output_file}")