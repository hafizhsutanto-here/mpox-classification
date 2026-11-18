import os
import sys
import platform
import importlib

print("=== SPESIFIKASI SISTEM & ENVIRONMENT ===")
print(f"OS            : {platform.system()} {platform.release()}")
print(f"Python        : {sys.version.split()[0]}")
print(f"CPU Cores     : {os.cpu_count()}")

print("\n=== CEK PYTORCH & GPU (CUDA) ===")
try:
    import torch
    print(f"PyTorch       : {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA Tersedia : {cuda_ok}")
    if cuda_ok:
        print(f"CUDA Version  : {torch.version.cuda}")
        print(f"Nama GPU      : {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"Total VRAM    : {vram_gb:.2f} GB")
    else:
        print("Catatan       : Hanya CPU yang terdeteksi. Jika punya GPU NVIDIA, install PyTorch CUDA:")
        print("                pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
except ImportError:
    print("PyTorch       : BELUM TERINSTALL (jalankan: pip install -r requirements.txt)")

print("\n=== CEK LIBRARY WAJIB ===")
libs = ["torchvision", "cv2", "sklearn", "matplotlib", "seaborn", "pandas", "kagglehub", "timm", "tqdm"]
for lib in libs:
    try:
        importlib.import_module(lib)
        print(f"  [OK] {lib}")
    except ImportError:
        print(f"  [MISSING] {lib}")
