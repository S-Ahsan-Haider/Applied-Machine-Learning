import torch

print("PyTorch version:", torch.__version__)
print("Compiled with CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU detected:", torch.cuda.get_device_name(0))
    
    # Minimal test: move a tensor to GPU and back
    
    x = torch.rand(3, 3).cuda()
    print("Tensor on GPU:", x.device)
    print("Sum:", x.sum().item())
