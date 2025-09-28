import torch

# testin CUDA's avalability

print(torch.cuda.is_available())                 

import torchvision
import torchvision.transforms as t

transform = t.Compose([
    t.ToTensor(),
    t.Normalize((0.5,),(0.5,))
])


# Loading and preprocessing the datasets

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True)


testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False)


# Building the Neural Network

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3*32*32, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 3*32*32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
# Training the loop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

moda = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(moda.parameters(), lr=0.001)

for epoch in range(5):
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = moda(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss: .3f}")

# Evaluate the model

correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = moda(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")

# Save and load the model

torch.save(moda.state_dict(), 'cifar_model.pth')

# To load:

# model.load_state_dict(torch.load('cifar_model.pth'))





