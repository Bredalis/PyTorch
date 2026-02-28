
import torch
import matplotlib.pyplot as plt

from mnist_dataset_exploration import test
from neural_network_model import model


def predict(model, image):
    model.eval()

    with torch.no_grad():
        logits = model(image)
        y_pred = logits.argmax(1).item()

    # Show image and predicted category
    plt.imshow(image.cpu().squeeze(), cmap="gray")
    plt.title(f"Predicted category: {y_pred}")
    plt.axis("off")
    plt.show()

    return y_pred


# Take an image from the test set
image, label = test[1235]

# Generate prediction
prediction = predict(model, image)

print(f"Real label: {label}")
print(f"Predicted label: {prediction}")
