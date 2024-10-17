import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from generated_mlp_minimal import Mlpminimal

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

def calc_entropy(degrees, weight_adjacency_tensor):
    laplacian = degrees - (weight_adjacency_tensor/torch.linalg.norm(weight_adjacency_tensor))
    aggr_laplacian = (laplacian[:, :, 0].detach() +
                      laplacian[:, :, 1].detach() +
                      laplacian[:, :, 2].detach())
    norm_laplacian = aggr_laplacian
    lambdas = torch.linalg.eigvals(norm_laplacian)
    algebr_entropy = 0.0
    for l in lambdas:
        algebr_entropy += -l * torch.log(l)
    return algebr_entropy, laplacian


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


def main():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

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
    model = Mlpminimal()
    print(model)
    #criterion = nn.CrossEntropyLoss()  # Mivel osztályozási feladat
    criterion = EntropyLoss(nn.CrossEntropyLoss())  # Mivel osztályozási feladat
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50

    n = 31
    # Adjacency tensor for weights (layers)
    weight_adjacency_tensor = torch.zeros(n, n, 3)
    degrees = torch.zeros(n, n, 3)
    degrees[0:4, 0:4, 0] = 16 * torch.diag(torch.ones(4))
    degrees[4:20, 4:20, 0] = 4 * torch.diag(torch.ones(16))

    degrees[4:20, 4:20, 1] = 8 * torch.diag(torch.ones(16))
    degrees[20:28, 20:28, 1] = 16 * torch.diag(torch.ones(8))

    degrees[20:28, 20:28, 2] = 3 * torch.diag(torch.ones(8))
    degrees[28:31, 28:31, 2] = 8 * torch.diag(torch.ones(3))

    degrees /= torch.linalg.norm(degrees)

    entropies = []
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
    model.eval()
    plt.plot(entropies)

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on test data: {100 * correct / total:.2f}%')

    algebr_entropy, laplacian = calc_entropy(degrees, weight_adjacency_tensor)

    print(algebr_entropy)
    plt.matshow(weight_adjacency_tensor[:, :, 0].detach().numpy())
    plt.matshow(weight_adjacency_tensor[:, :, 1].detach().numpy())
    plt.matshow(weight_adjacency_tensor[:, :, 2].detach().numpy())

    plt.matshow(laplacian[:, :, 0].detach().numpy() +
                laplacian[:, :, 1].detach().numpy() +
                laplacian[:, :, 2].detach().numpy())
    plt.show()


if __name__ == "__main__":
    main()
