import torch
import torch.nn as nn
import torch.optim as optim
from generated_mlp import Mlp

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

def main():
    # Betöltjük az IRIS adatokat
    iris = datasets.load_iris()
    X = iris.data  # Bemeneti adatok (4 jellemző)
    y = iris.target  # Célértékek (osztálycímkék)

    # Az adatok normalizálása (StandardScaler használatával)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Adatok betöltése PyTorch tensorokba
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    print(X_tensor.shape, y_tensor.shape)

    # Adatok felosztása tréning és teszt adatokra
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    # PyTorch DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = Mlp()
    print(model)
    # Veszteségfüggvény és optimalizáló
    criterion = nn.CrossEntropyLoss()  # Mivel osztályozási feladat
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Tanítási ciklus
    num_epochs = 50

    for epoch in range(num_epochs):
        model.train()  # Tanítási módba kapcsoljuk a modellt
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Gradiensek nullázása
            outputs = model(inputs)  # Előrecsatolás
            loss = criterion(outputs, labels)  # Veszteség kiszámítása
            loss.backward()  # Backpropagation
            optimizer.step()  # Súlyok frissítése

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
    # Tesztelés
    model.eval()  # Tesztelési módba kapcsoljuk a modellt
    correct = 0
    total = 0

    with torch.no_grad():  # Tesztelés során nem számolunk gradienseket
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on test data: {100 * correct / total:.2f}%')


if __name__ == "__main__":
    main()
