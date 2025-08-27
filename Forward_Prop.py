
import torch
import matplotlib.pyplot as plt
from Dataset import train
from Red_Neuronal import modelo

# Extraer una imagen y su categoría del set de entrenamiento
img, lbl = train[200]

print(type(img))
print(type(lbl))

# Convertir 'lbl' a Tensor usando 'tensor', definir tamaño 
# igual a 1 (1 dato) con 'reshape'
lbl = torch.tensor(lbl).reshape(1)
print(type(lbl))

# Forward propagetion
logits = modelo(img)
print(logits)

# Categoría predicha
y_pred = logits.argmax(1)

# Mostrar la imagen original
plt.imshow(img.cpu().squeeze(), cmap = "gray");

# Comparar la categoría predicha con la categoría real
print(f"Logits: {logits}")
print(f"Categoría predicha: {y_pred[0]}")
print(f"Categoría real: {lbl[0]}")