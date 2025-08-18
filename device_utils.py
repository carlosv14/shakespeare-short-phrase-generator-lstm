import torch

def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    return device

def print_device_info():
    print("PyTorch Device Information:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  MPS available: {torch.backends.mps.is_available()}")
    print(f"  MPS built: {torch.backends.mps.is_built()}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    if torch.backends.mps.is_available():
        print(f"  MPS device: {torch.device('mps')}")
    
    if torch.cuda.is_available():
        print(f"  CUDA device count: {torch.cuda.device_count()}")
        print(f"  Current CUDA device: {torch.cuda.current_device()}")
        print(f"  CUDA device name: {torch.cuda.get_device_name()}")
    
    print(f"  CPU device: {torch.device('cpu')}")
    print()

