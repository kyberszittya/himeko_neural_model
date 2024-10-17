import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from generated_mlp_minimal import Mlpminimal

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from generated_mnist_mlp_network import Mnist_mlp_network


def main():
    batch_size = 64

    # Define the transformations to apply on the MNIST images (normalize them)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    # Use DataLoader to iterate through the data in batches
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    input_size = 28 * 28

    model = Mnist_mlp_network()
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50



    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_loss = 0
        for inputs, labels in train_loader:
            # Flatten the images into vectors of size 28*28
            inputs = inputs.view(-1, input_size)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass (calculate gradients)
            loss.backward()

            # Update the weights
            optimizer.step()

            total_loss += loss.item()

        # Print the loss after every epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

    # Testing the model
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            # Flatten the images into vectors of size 28*28
            inputs = inputs.view(-1, input_size)

            # Forward pass to get predictions
            outputs = model(inputs)

            # Get the predicted class (highest score in the output layer)
            _, predicted = torch.max(outputs.data, 1)

            # Calculate the number of correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print the accuracy
    print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')


if __name__ == "__main__":
    main()
