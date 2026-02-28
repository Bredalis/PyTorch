
import torch


# Manually create a tensor
array = [[2, 3, 4], [1, 5, 6]]
tensor_1 = torch.tensor(array)  # 2D tensor

print(tensor_1)


# Check current device
print(f"Initial device: {tensor_1.device}")


# Select device (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"


# Move tensor to selected device
tensor_1 = tensor_1.to(device)

print(f"Device in use: {tensor_1.device}")


# Tensor shape
print(f"Tensor shape: {tensor_1.shape}")
