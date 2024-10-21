import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

input_size = 28 * 28
hidden_size = 128
output_size = 10
weight_decay = 0.0001
learning_rate = 0.04

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

train_dataset = datasets.MNIST(root='./mnist_data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./mnist_data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


def train_model(model, train_loader, test_loader, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(1, num_epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            #output = model(data.view(-1, 28*28))  # Cant do this with model3
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        test_loss, accuracy = validate(model, test_loader, criterion)
        print(f'{epoch} & {accuracy:.2f}\\%')
    
    return model

def validate(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad(): # disable gradient calculation for validation
        for data, target in test_loader:
            #output = model(data.view(-1, 28*28)) # Cant do this with model3
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    return test_loss, accuracy


model1 = nn.Sequential(
    nn.Linear(28*28, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size)
)


#train_model(model1, train_loader, test_loader, num_epochs=10)

model2 = nn.Sequential(
    nn.Linear(28*28, 500),
    nn.ReLU(),
    nn.Linear(500, 300),
    nn.ReLU(),
    nn.Linear(300, 10),
)

#train_model(model2, train_loader, test_loader, num_epochs=40)

model3 = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Flatten(),
    nn.Linear(64 * 7 * 7, 128),
    nn.ReLU(),

    nn.Linear(128, 10)
)

train_model(model3, train_loader, test_loader, num_epochs=40)