import torch

print("GPU Name:", torch.cuda.get_device_name(0))
print("GPU Count:", torch.cuda.device_count())
print("Current Device:", torch.cuda.current_device())
