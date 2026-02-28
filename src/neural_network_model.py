
from torch import nn

from tensor_device_demo import device


"""
Create the Neural Network as a subclass of nn.Module.

Two methods are always added:
1. __init__: defines the network architecture
2. forward: defines how each prediction is generated
"""


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Flatten input images
        self.flatten = nn.Flatten()

        # Sequential layers
        self.network = nn.Sequential(
            nn.Linear(28 * 28, 15),  # Input layer + hidden layer
            nn.ReLU(),              # Activation function
            nn.Linear(15, 10),      # Output layer (no activation)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits


# Instantiate model and move to device
model = NeuralNetwork().to(device)
print(model)

# Count trainable parameters
total_parameters = sum(
    p.numel() for p in model.parameters()
)

print("Number of trainable parameters:", total_parameters)
