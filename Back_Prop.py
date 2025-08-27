
import torch
from torch import nn
from Red_Neuronal import modelo
from Forward_Prop import logits, lbl

# Pasos para la propagación hacia atras

# 0. Pérdida y optimizador
fn_perdida = nn.CrossEntropyLoss()
optimizador = torch.optim.SGD(modelo.parameters(), lr = 0.2) # Se ponen acá los parámetros para que se actualícen

# 1. Calcular pérdida
loss = fn_perdida(logits, lbl)
print(loss)

# 2. Calcular los gradientes de la pérdida
loss.backward()

# 3. Actualizar los parámetros del modelo
optimizador.step()
optimizador.zero_grad()