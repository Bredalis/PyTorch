
import torch
import matplotlib.pyplot as plt

from mnist_dataset_exploration import train
from neural_network_model import model


# Extract an image and its label from the training set
img, label = train[200]

print(type(img))
print(type(label))


# Convert label to tensor and reshape to size 1
label = torch.tensor(label).reshape(1)
print(type(label))


# Forward propagation
logits = model(img)
print(logits)


# Predicted class
y_pred = logits.argmax(1)


# Show original image
plt.imshow(img.cpu().squeeze(), cmap="gray")
plt.axis("off")
plt.show()


# Compare predicted vs real label
print(f"Logits: {logits}")
print(f"Predicted label: {y_pred[0]}")
print(f"Real label: {label[0]}")
