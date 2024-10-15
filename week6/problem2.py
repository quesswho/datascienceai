import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# Hyperparameters
input_size = 28 * 28  # 784 for MNIST images
hidden_size = 128     # Size of the hidden layer
output_size = 10      # 10 classes for digits 0-9
learning_rate = 0.01  # Learning rate for SGD
num_epochs = 10       # Number of epochs
batch_size = 64       # Batch size

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

train_dataset = datasets.MNIST(root='./mnist_data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./mnist_data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize weights and biases for the layers
W1 = torch.randn(input_size, hidden_size, requires_grad=True)  # Weight for input to hidden layer
b1 = torch.zeros(hidden_size, requires_grad=True)              # Bias for hidden layer
W2 = torch.randn(hidden_size, output_size, requires_grad=True) # Weight for hidden to output layer
b2 = torch.zeros(output_size, requires_grad=True)              # Bias for output layer

# Loss function
criterion = nn.CrossEntropyLoss()

# Use SGD optimizer
optimizer = optim.SGD([W1, b1, W2, b2], lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images = images.view(-1, input_size)  # Flatten the images

        # Forward pass
        hidden_layer = torch.matmul(images, W1) + b1  # Linear transformation
        hidden_layer = torch.relu(hidden_layer)       # ReLU activation
        outputs = torch.matmul(hidden_layer, W2) + b2 # Linear transformation to output layer

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()  # Zero the gradients
        loss.backward()        # Backpropagation
        optimizer.step()       # Update weights and biases

    # Validation accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, input_size)
            hidden_layer = torch.matmul(images, W1) + b1
            hidden_layer = torch.relu(hidden_layer)
            outputs = torch.matmul(hidden_layer, W2) + b2
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.4f}')

# Final evaluation on test set
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(-1, input_size)
        hidden_layer = torch.matmul(images, W1) + b1
        hidden_layer = torch.relu(hidden_layer)
        outputs = torch.matmul(hidden_layer, W2) + b2
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

final_accuracy = correct / total
print(f'Final Test Accuracy: {final_accuracy:.4f}')