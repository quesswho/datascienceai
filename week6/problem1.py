import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Define transformations to apply to the images
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1]
])

# Download the MNIST dataset
train_dataset = datasets.MNIST(root='./mnist_data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./mnist_data', train=False, download=True, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Function to display images
def show_images(images, labels):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')  # Convert to 2D for visualization
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Get a batch of training images
images, labels = next(iter(train_loader))

# Show images from the training set
show_images(images.numpy(), labels.numpy())

images, labels = next(iter(test_loader))

# Show images from the test set
show_images(images.numpy(), labels.numpy())

# Verify the dimensions and normalization
print(f"Image shape: {images.shape}")  # Should be [64, 1, 28, 28]
print(f"Pixel value range: [{images.min().item()}, {images.max().item()}]")  # Should be around [-1, 1]
