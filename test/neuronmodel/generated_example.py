import gc
import os

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from generated_mlp_minimal import Mlpminimal

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader


def calc_entropy(degrees, weight_adjacency_tensor):
    weight_abs = torch.abs(weight_adjacency_tensor)

    aggr_adj = (weight_abs[:, :, 0].detach() +
                      weight_abs[:, :, 1].detach() +
                      weight_abs[:, :, 2].detach())
    degrees_ = (degrees[:, :, 0].detach() +
                degrees[:, :, 1].detach() +
                degrees[:, :, 2].detach())
    degree_norm = torch.linalg.norm(degrees_)
    max_degree = torch.max(degrees)
    norm_laplacian = ((degrees_/max_degree) - aggr_adj) / (degree_norm)
    # Check if symmetric
    lambdas = torch.linalg.eigvals(norm_laplacian)

    algebr_entropy = 0.0
    for l in lambdas:
        algebr_entropy += -l * torch.log2(l)
    return algebr_entropy, norm_laplacian


class EntropyLoss(nn.Module):
    def __init__(self, base_loss, lambda_entropy=0.1):
        super(EntropyLoss, self).__init__()
        self.base_loss = base_loss
        self.lambda_entropy = lambda_entropy

    def forward(self, outputs, targets, degrees, weights_tensor):
        base_loss_value = self.base_loss(outputs, targets)
        entropy, _ = calc_entropy(degrees, weights_tensor)

        total_loss = base_loss_value + self.lambda_entropy * entropy
        return total_loss


def model_mlp_eval(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct, total


def model_normal_training(train_loader, test_loader, num_epochs):
    model = Mlpminimal()
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    accs = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
        correct, total = model_mlp_eval(model, test_loader)
        acc = correct / total
        accs.append(acc)
        print(f'Accuracy on test data: {100 * acc:.2f}%')
    del model
    del optimizer
    return accs


def main():
    iris = datasets.load_iris()
    iris = datasets.make_classification(
        n_samples=1150,       # Same as the Iris dataset
        n_features=4,        # Number of features (sepal length, width, etc.)
        n_informative=4,     # All features are informative
        n_redundant=0,       # No redundant features
        n_classes=3,         # Number of classes
        n_clusters_per_class=1,  # Similar to Iris dataset
        class_sep=1.5,       # Control separability of classes
        random_state=42      # For reproducibility
    )

    #X = iris.data
    #y = iris.target
    X, y = iris

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    print(X_tensor.shape, y_tensor.shape)

    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    #model = Mlp()



    num_epochs = 1500

    n = 31
    # Adjacency tensor for weights (layers)

    degrees = torch.zeros(n, n, 3)
    degrees[0:4, 0:4, 0] = 16 * torch.diag(torch.ones(4))
    degrees[4:20, 4:20, 0] = 4 * torch.diag(torch.ones(16))

    degrees[4:20, 4:20, 1] = 8 * torch.diag(torch.ones(16))
    degrees[20:28, 20:28, 1] = 16 * torch.diag(torch.ones(8))

    degrees[20:28, 20:28, 2] = 3 * torch.diag(torch.ones(8))
    degrees[28:31, 28:31, 2] = 8 * torch.diag(torch.ones(3))
    # Normal training process
    cnt_training = 10
    start_index = 0
    for i in range(start_index, cnt_training):
        weight_adjacency_tensor = torch.zeros(n, n, 3)
        model = Mlpminimal()
        print(model)
        criterion = EntropyLoss(nn.CrossEntropyLoss())
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        accs_normal = model_normal_training(train_loader, test_loader, num_epochs)

        entropies = []
        accs = []
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)

                weight_adjacency_tensor[4:20, 0:4, 0] = torch.abs(model.input_layer.weight)
                weight_adjacency_tensor[0:4, 4:20, 0] = torch.abs(model.input_layer.weight.T)
                # Next layer
                weight_adjacency_tensor[20:28, 4:20, 1] = torch.abs(model.hidden_layer.weight)
                weight_adjacency_tensor[4:20, 20:28, 1] = torch.abs(model.hidden_layer.weight.T)
                # Next layer
                weight_adjacency_tensor[28:31, 20:28, 2] = torch.abs(model.output_layer.weight)
                weight_adjacency_tensor[20:28, 28:31, 2] = torch.abs(model.output_layer.weight.T)

                loss = criterion(outputs, labels, degrees, weight_adjacency_tensor)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            algebr_entropy, laplacian = calc_entropy(degrees, weight_adjacency_tensor)

            print(algebr_entropy)
            entropies.append(algebr_entropy)

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
            correct, total = model_mlp_eval(model, test_loader)
            acc = correct / total
            accs.append(acc)
            print(f'Accuracy on test data: {100 * acc:.2f}%')

        plt.plot(entropies)

        algebr_entropy, laplacian = calc_entropy(degrees, weight_adjacency_tensor)

        print(algebr_entropy)
        # Generate folder "entropy_data" if it does not exist (OK)
        if not os.path.exists("entropy_data"):
            os.makedirs("entropy_data_1000")
        # Save entropy, accuracy and entropy accuracy
        with open(f"entropy_data/entropy_acc_{i}_steps_1000.txt", "w") as f:
            f.write("Entropy,NormalAccuracy,EntropyAccuracy\n")
            for row,_ in enumerate(accs):
                f.write(f"{entropies[row].real},{accs_normal[row]},{accs[row]}\n")
        del model
        del optimizer
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()
        plt.show()
        plt.plot(accs_normal, label='Normal')
        plt.plot(accs, label='With Entropy')
        plt.legend()
        plt.grid()
        plt.show()







if __name__ == "__main__":
    main()
