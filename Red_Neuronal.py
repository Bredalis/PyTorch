
from torch import nn
from Tensores import device

"""
Crear la Red Neuronal como una subclase de nn.Module
Siempre se añaden dos métodos a esta subclase
1. Método 'init': define la arquitectura de la red
2. Método 'forward': define cómo será generada cada predicción
"""

class RedNeuronal(nn.Module):
    # 1. Método 'init'
    def __init__(self):
        super().__init__()

        # Y agregar secuencialmente las capas
        self.aplanar = nn.Flatten() # Aplanar imágenes de entrada
        self.red = nn.Sequential(
            nn.Linear(28 * 28, 15), # Capa de entrada + capa oculta
            nn.ReLU(), # Función de activación capa oculta
            nn.Linear(15, 10), # Capa de salida sin activación
        )

    # 2. Método 'forward' (x = dato de entrada)
    def forward(self, x):
        # Definir secuencialmente las operaciones a aplicar
        x = self.aplanar(x) # Aplanar dato
        logits = self.red(x) # Generar predicción

        return logits

# Mostrar información del modelo
modelo = RedNeuronal().to(device)
print(modelo)

total_parametros = sum(p.numel() for p in modelo.parameters())
print("Número de parámentros a entrenar: ", total_parametros)