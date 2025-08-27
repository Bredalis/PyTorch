
import torch
from torchvision import datasets
# Para convertir los datos a Tensores
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# Crear el directorio 'datos'
data_mnist = datasets.MNIST(
	root = "datos", # Carpeta donde se almacenará
	train = True, # True: 60.000 imágenes, False: 10.000 imágenes
	download = True,
	transform = ToTensor() # Convertir imágenes a tensores
)

print(f"El set de datos (Información):\n {data_mnist}")

# Mostrar las imágenes
figura = plt.figure(figsize = (8, 8))
filas, columnas = 3, 3

for i in range(1, columnas * filas + 1):
  # Escoger una imagen aleatoria
  ejemplo_idx = torch.randint(len(data_mnist), size = (1,)).item()

  # Extraer imagen y categoría
  img, label = data_mnist[ejemplo_idx]

  # Dibujar
  figura.add_subplot(filas, columnas, i)
  plt.title(str(label)) # Categoría
  plt.axis("off")
  plt.imshow(img.squeeze(), cmap = "gray") # Mostrar la imagen

plt.show()

# Cacterísticas de una imagen
print(f"Tipo de dato imagen: {type(img)}")
print(f"Tamaño imagen: {img.shape}")
print(f"Mínimo y máximo imagen: {img.min()}, {img.max()}")
print(f"Tipo de dato categoría: {type(label)}")

torch.manual_seed(123)

# División del dataset
train, val, test = torch.utils.data.random_split(
    data_mnist, [.8, .1, .1]
)

# Verificar tamaños 
print("\nTamaño de los sets\n")
print(f"Entrenamiento: {len(train)}")
print(f"Validación: {len(val)}")
print(f"Prueba: {len(test)}")

# Verificar el tipo de dato 
print("\nTipo de dato de los sets\n")
print(f"Entrenamiento: {type(train)}")
print(f"Validación: {type(val)}")
print(f"Prueba: {type(test)}")