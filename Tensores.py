
import torch

# Crear un tensor de forma manual
arreglo = [[2, 3, 4], [1, 5, 6]]
tensor_1 = torch.tensor(arreglo) # Arreglo 2D
print(tensor_1)

tensor_1.device # Saber el procesador que usa

# Cambiar el procesador
device = (
    "cuda" if torch.cuda.is_available() else "cpu"
)

tensor_1.to(device)
print(f"Procesador usado: {tensor_1.device}")

# Tamaño del tensor
print(tensor_1.shape)