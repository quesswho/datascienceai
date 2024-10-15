import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# Hyperparameters
input_size = 28 * 28  # 784 for MNIST images
hidden_size1 = 500    # Size of the first hidden layer
hidden_size2 = 300    # Size of the second hidden layer
output_size = 10      # 10 classes for digits 0-9
learning_rate = 0.005  # Learning rate for SGD
num_epochs = 40       # Number of epochs
batch_size = 64       # Batch size
weight_decay = 1e-6   # L2 regularization factor (weight decay)

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
W1 = torch.randn(input_size, hidden_size1, requires_grad=True)  # Weight for input to first hidden layer
b1 = torch.zeros(hidden_size1, requires_grad=True)              # Bias for first hidden layer
W2 = torch.randn(hidden_size1, hidden_size2, requires_grad=True) # Weight for first hidden to second hidden layer
b2 = torch.zeros(hidden_size2, requires_grad=True)              # Bias for second hidden layer
W3 = torch.randn(hidden_size2, output_size, requires_grad=True) # Weight for second hidden to output layer
b3 = torch.zeros(output_size, requires_grad=True)               # Bias for output layer

# Loss function
criterion = nn.CrossEntropyLoss()

# Use SGD optimizer with L2 regularization (weight_decay)
optimizer = optim.SGD([W1, b1, W2, b2, W3, b3], lr=learning_rate, weight_decay=weight_decay)

# Training loop
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images = images.view(-1, input_size)  # Flatten the images

        # Forward pass
        hidden_layer1 = torch.matmul(images, W1) + b1  # Linear transformation
        hidden_layer1 = torch.relu(hidden_layer1)      # ReLU activation

        hidden_layer2 = torch.matmul(hidden_layer1, W2) + b2  # Linear transformation
        hidden_layer2 = torch.relu(hidden_layer2)             # ReLU activation

        outputs = torch.matmul(hidden_layer2, W3) + b3  # Linear transformation to output layer

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
            hidden_layer1 = torch.matmul(images, W1) + b1
            hidden_layer1 = torch.relu(hidden_layer1)
            hidden_layer2 = torch.matmul(hidden_layer1, W2) + b2
            hidden_layer2 = torch.relu(hidden_layer2)
            outputs = torch.matmul(hidden_layer2, W3) + b3
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'{epoch + 1} {accuracy*100:.4f} \%')

# Final evaluation on test set
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(-1, input_size)
        hidden_layer1 = torch.matmul(images, W1) + b1
        hidden_layer1 = torch.relu(hidden_layer1)
        hidden_layer2 = torch.matmul(hidden_layer1, W2) + b2
        hidden_layer2 = torch.relu(hidden_layer2)
        outputs = torch.matmul(hidden_layer2, W3) + b3
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

final_accuracy = correct / total
print(f'Final Test Accuracy: {final_accuracy:.4f}')