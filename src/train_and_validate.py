
import torch
from torch import nn
from torch.utils.data import DataLoader

from mnist_dataset_exploration import train, val
from neural_network_model import model
from tensor_device_demo import device


# Configuration
BATCH_SIZE = 1000
LEARNING_RATE = 0.1
EPOCHS = 10


# DataLoaders
train_loader = DataLoader(
    dataset=train,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    dataset=val,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# Loss and Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
)


# Training Loop
def train_loop(dataloader, model, loss_function, optimizer):
    train_size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.train()

    total_loss = 0
    total_accuracy = 0

    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Forward pass
        logits = model(x)

        # Backward pass
        loss = loss_function(logits, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        total_accuracy += (
            (logits.argmax(1) == y)
            .type(torch.float)
            .sum()
            .item()
        )

        if batch_idx % 10 == 0:
            processed = batch_idx * BATCH_SIZE
            print(
                f"\tLoss: {loss.item():>7f} "
                f"[{processed:>5d}/{train_size:>5d}]"
            )

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / train_size

    print("\tAverage accuracy/loss:")
    print(
        f"\t\tTraining: "
        f"{(100 * avg_accuracy):>0.1f}% / {avg_loss:>8f}"
    )


# Validation Loop
def val_loop(dataloader, model, loss_function):
    val_size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()

    total_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            logits = model(x)

            total_loss += loss_function(logits, y).item()
            total_accuracy += (
                (logits.argmax(1) == y)
                .type(torch.float)
                .sum()
                .item()
            )

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / val_size

    print(
        f"\t\tValidation: "
        f"{(100 * avg_accuracy):>0.1f}% / {avg_loss:>8f}\n"
    )


# Training Process
for epoch in range(EPOCHS):
    print(
        f"Epoch {epoch + 1}/{EPOCHS}\n"
        "-------------------------------"
    )

    train_loop(train_loader, model, loss_fn, optimizer)
    val_loop(val_loader, model, loss_fn)

print("Done! The model has been trained.")
