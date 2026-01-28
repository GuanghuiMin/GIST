import torch
import os
import sys
import glob


def check_file_dimension(file_path):
    print(f"Checking: {os.path.basename(file_path)} ...")

    try:
        data = torch.load(file_path, map_location='cpu')

        if isinstance(data, torch.Tensor):
            print(f"Type: Tensor")
            print(f"Shape: {data.shape}")
            print(f"DT Dtype: {data.dtype}")
            print(f"Projection Dim (Last Dim): {data.shape[-1]}")

        elif isinstance(data, dict):
            print(f"Type: Dict (Contains multiple keys)")
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    print(f"      - Key '{k}': Shape {v.shape}")
                else:
                    print(f"      - Key '{k}': {type(v)}")
        else:
            print(f"Type: {type(data)}")

    except Exception as e:
        print(f"Error loading file: {e}")
    print("-" * 40)


def main():
    # 1. python check_dims.py path/to/your/file.pt
    # 2. python check_dims.py path/to/folder/
    if len(sys.argv) < 2:
        print("Usage: python check_dims.py <file_path_or_dir>")
        # target_path = "./projections"
        target_path = input("Please input your .pt file or directory").strip()
    else:
        target_path = sys.argv[1]

    if os.path.isfile(target_path):
        check_file_dimension(target_path)
    elif os.path.isdir(target_path):
        pt_files = glob.glob(os.path.join(target_path, "*.pt"))
        if not pt_files:
            print("no .pt files found in the directory.")
        for f in sorted(pt_files):
            check_file_dimension(f)
    else:
        print("No such file or directory found.")


if __name__ == "__main__":
    main()