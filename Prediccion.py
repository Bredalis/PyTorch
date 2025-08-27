
import maplotlib.pyplot as plt
from Dataset import test
from Red_Neuronal import modelo

def predecir(model, img):
    # Generar predicción
    logits = model(img)
    y_pred = logits.argmax(1).item()

    # Mostrar imagen original y categoría predicha
    plt.imshow(img.cpu().squeeze(), cmap = "gray")
    plt.title(f"Categoría predicha: {y_pred}")

# Tomar una imagen del set de prueba
img, lbl = test[1235]

# Y generar la predicción
predecir(modelo, img)