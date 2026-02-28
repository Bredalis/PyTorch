
import torch
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor


# Create the 'data' directory and load MNIST
data_mnist = datasets.MNIST(
    root="data",  # Folder where it will be stored
    train=True,  # True: 60,000 images | False: 10,000 images
    download=True,
    transform=ToTensor(),  # Convert images to tensors
)

print(f"Dataset information:\n{data_mnist}")


# Display images
figure = plt.figure(figsize=(8, 8))
rows, columns = 3, 3

for i in range(1, columns * rows + 1):
    # Select a random image
    example_idx = torch.randint(
        len(data_mnist),
        size=(1,),
    ).item()

    # Extract image and label
    img, label = data_mnist[example_idx]

    # Plot
    figure.add_subplot(rows, columns, i)
    plt.title(str(label))  # Category
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")  # Show image

plt.show()


# Image characteristics
print(f"Image data type: {type(img)}")
print(f"Image shape: {img.shape}")
print(f"Image min and max: {img.min()}, {img.max()}")
print(f"Label data type: {type(label)}")


torch.manual_seed(123)

# Dataset split
train, val, test = torch.utils.data.random_split(
    data_mnist,
    [0.8, 0.1, 0.1],
)

# Verify sizes
print("\nDataset sizes\n")
print(f"Training: {len(train)}")
print(f"Validation: {len(val)}")
print(f"Test: {len(test)}")

# Verify data types
print("\nDataset types\n")
print(f"Training: {type(train)}")
print(f"Validation: {type(val)}")
print(f"Test: {type(test)}")
