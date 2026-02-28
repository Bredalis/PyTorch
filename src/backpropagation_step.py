
import torch
from torch import nn

from neural_network_model import model
from single_prediction_test import logits, label


# Steps for backpropagation

# 0. Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.2,
)  # Parameters are passed here so they can be updated

# 1. Compute loss
loss = loss_fn(logits, label)
print(loss)

# 2. Compute gradients
loss.backward()

# 3. Update model parameters
optimizer.step()
optimizer.zero_grad()
