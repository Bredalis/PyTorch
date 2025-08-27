
import torch
from torch import nn
from torch.utils.data import DataLoader
from Dataset import train, val
from Red_Neuronal import modelo
from Tensores import device

# Definir el tamaño del lote
TAM_LOTE = 1000 # batch size

# Crear los 'dataloaders' para los sets de entrenamiento y validación
train_loader = DataLoader(
    dataset = train,
    batch_size = TAM_LOTE,
    shuffle = True # Mezclar los datos aleatoriamente al crear cada lote
)

val_loader = DataLoader(
    dataset = val,
    batch_size = TAM_LOTE,
    shuffle = False
)

# Hiperparámetros
TASA_APRENDIZAJE = 0.1 # learning rate (0.1)
EPOCHS = 10 # Número de iteraciones de entrenamiento

# Función de pérdida y optimizador
fn_perdida = nn.CrossEntropyLoss()
optimizador = torch.optim.SGD(modelo.parameters(), lr = TASA_APRENDIZAJE)

def train_loop(dataloader, model, loss_fn, optimizer):
    # Cantidad de datos de entrenamiento y cantidad de lotes
    train_size = len(dataloader.dataset)
    nlotes = len(dataloader)

    # Indicarle a Pytorch que entrenaremos el modelo
    model.train()

    # Inicializar acumuladores pérdida y exactitud
    perdida_train, exactitud = 0, 0

    # Presentar los datos al modelo por lotes (de tamaño TAM_LOTE)
    for nlote, (X, y) in enumerate(dataloader):
        # Mover "X" y "y" a la GPU
        X, y = X.to(device), y.to(device)

        # Forward propagation
        logits = model(X)

        # Backpropagation
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Acumular valores de pérdida y exactitud
        # perdida_train <- perdida_train + perdida_actual
        # exactitud <- exactitud + numero_aciertos_actuales
        perdida_train += loss.item()
        exactitud += (logits.argmax(1)==y).type(torch.float).sum().item()

        # Imprimir en pantalla la evolución del entrenamiento (cada 10 lotes)
        if nlote % 10 == 0:
            # Obtener el valor de la pérdida (loss) y el número de datos procesados (ndatos)
            ndatos = nlote*TAM_LOTE

            # E imprimir en pantalla
            print(f"\tPérdida: {loss.item():>7f}  [{ndatos:>5d}/{train_size:>5d}]")

    # Al terminar de presentar todos los datos al modelo, promediar pérdida y exactitud
    perdida_train /= nlotes # Pérdida promedio = pérdida acumulada / número de lotes
    exactitud /= train_size # Exactitud promedio = exactitud acumulada / número de datos

    # E imprimir información
    print(f'\tExactitud/pérdida promedio:')
    print(f'\t\tEntrenamiento: {(100*exactitud):>0.1f}% / {perdida_train:>8f}')


def val_loop(dataloader, model, loss_fn):
    # Cantidad de datos de validación y cantidad de lotes
    val_size = len(dataloader.dataset)
    nlotes = len(dataloader)

    # Indicarle a Pytorch que validaremos el modelo
    model.eval()

    # Inicializar acumuladores pérdida y exactitud
    perdida_val, exactitud = 0, 0

    # Evaluar (generar predicciones) usando "no_grad"
    with torch.no_grad():
        for X, y in dataloader:
            # Mover "X" y "y" a la GPU
            X, y = X.to(device), y.to(device)

            # Propagación hacia adelante (predicciones)
            logits = model(X)

            # Acumular valores de pérdida y exactitud
            perdida_val += loss_fn(logits, y).item()
            exactitud += (logits.argmax(1) == y).type(torch.float).sum().item()

    # Tras generar las predicciones calcular promedios de pérdida y exactitud
    perdida_val /= nlotes
    exactitud /= val_size

    # E imprimir en pantalla
    print(f"\t\tValidación: {(100*exactitud):>0.1f}% / {perdida_val:>8f} \n")

for t in range(EPOCHS):
    print(f"Iteración {t+1}/{EPOCHS}\n-------------------------------")
    # Entrenar
    train_loop(train_loader, modelo, fn_perdida, optimizador)
    # Validar
    val_loop(val_loader, modelo, fn_perdida)
print("Listo, el modelo ha sido entrenado!")