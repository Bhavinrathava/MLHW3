import sys
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import datetime
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, labels = self.data[idx][1], self.data[idx][0]
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return features_tensor, labels_tensor.squeeze()
    
class CustomMnistDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, labels = self.data[idx][1], self.data[idx][0]
        features = [int(x) for x in features]
        labels = int(labels)
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return features_tensor, labels_tensor.squeeze()

def softmax(x):
    exp_x = torch.exp(x - torch.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum(dim=-1, keepdim=True)

def preprocess_data(data):
    # Preprocessing the Insurance data
    min_x1 = min_x2 = min_x3 = 100000
    max_x1 = max_x2 = max_x3 = -100000

    for row in data:
        features = row[1]
        min_x1 = min(min_x1, features[0])
        min_x2 = min(min_x2, features[1])
        min_x3 = min(min_x3, features[2])

        max_x1 = max(max_x1, features[0])
        max_x2 = max(max_x2, features[1])
        max_x3 = max(max_x3, features[2])

    for row in data:
        features = row[1]
        features[0] -= min_x1
        features[0] /= (max_x1 - min_x1)

        features[1] -= min_x2
        features[1] /= (max_x2 - min_x2)

        features[2] -= min_x3
        features[2] /= (max_x3 - min_x3)
        row[1] = features
    return 

class FFNN(nn.Module):
        
        def __init__(self, input_size, hidden_size, output_size):
            super(FFNN, self).__init__()
            self.linear1 = nn.Linear(input_size, hidden_size)
            self.sigmoid1 = nn.Sigmoid()
            self.linear2 = nn.Linear(hidden_size, output_size) 
            self.softmax = softmax
        def forward(self, x):

            
            x = self.linear1(x)
            x = self.sigmoid1(x)
            x = self.linear2(x)
            x = self.softmax(x)
            return x       

def trainModel(train_loader, model, loss_func, optimizer, device):
  model.train()
  train_loss = []
  now = datetime.datetime.now()

  for batch_idx, (features, labels) in enumerate(train_loader):
    # make some predictions and get the error
    pred = model(features)
    loss = loss_func(pred, labels)

    # where the magic happens
    # backpropogation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch_idx % 50 == 0:
      loss, current = loss.item(), batch_idx * len(features)
      iters = 10 * len(features)
      then = datetime.datetime.now()
      iters /= (then - now).total_seconds()
      #print(f"loss: {loss:>6f} [{current:>5d}/{17000}] ({iters:.1f} its/sec)")
      now = then
      train_loss.append(loss)

  return sum(train_loss)/len(train_loss)

def testModel(dataset, model, loss_func):
  model.eval()
  test_loss = 0
  correct = 0
  predicted_labels = []
  true_labels = []
  with torch.no_grad():
    for batch_idx, (X, y) in enumerate(dataset):
      pred = model(X)
      test_loss += loss_func(pred, y).item()
      pred = pred.argmax(dim=1, keepdim=True)
      correct += pred.eq(y.view_as(pred)).sum().item()
      true_labels.extend(y.numpy())
      predicted_labels.extend(pred.numpy())
  test_loss /= len(dataset.dataset)

  #print(f'Validation set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(dataset.dataset)} ({100. * correct / len(dataset.dataset):.0f}%)\n')
  #print(f"Avg Loss: {test_loss:>8f}\n")

  return test_loss, 100. * correct / len(dataset.dataset)

def evaluate_model(testDataset, model,loss_func):
    model.eval()
    test_loss = 0
    correct = 0
    predicted_labels = []
    true_labels = []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(testDataset):
            pred = model(X)
            test_loss += loss_func(pred, y).item()
            pred = pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
            true_labels.extend(y.numpy())
            predicted_labels.extend(pred.numpy())
    test_loss /= len(testDataset.dataset)

    #print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(testDataset.dataset)} ({100. * correct / len(testDataset.dataset):.0f}%)\n')
    #print(f"Avg Loss: {test_loss:>8f}\n")
    
    # Calculate F1 Score
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    #print(f'F1 Score: {f1}')
    #print("\n ----------------------------------------- \n")
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    # Plot confusion matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('Actual Labels')
    plt.xlabel('Predicted Labels')
    plt.title('Confusion Matrix')
    plt.show()

    return test_loss, 100. * correct / len(testDataset.dataset)
                 
def read_insurability(file_name):
    
    count = 0
    data = []
    with open(file_name,'rt') as f:
        for line in f:
            if count > 0:
                line = line.replace('\n','')
                tokens = line.split(',')
                if len(line) > 10:
                    x1 = float(tokens[0])
                    x2 = float(tokens[1])
                    x3 = float(tokens[2])
                    if tokens[3] == 'Good':
                        cls = 0
                    elif tokens[3] == 'Neutral':
                        cls = 1
                    else:
                        cls = 2
                    data.append([[cls],[x1,x2,x3]])
            count = count + 1
    return(data)
               
def classify_insurability(device,preprocess=False, early_stopping=True):
    
    train = read_insurability('three_train.csv')
    valid = read_insurability('three_valid.csv')
    test = read_insurability('three_test.csv')
    
    epochs = 50
    LR = 0.05

    if(preprocess):
        preprocess_data(train)
        preprocess_data(valid)
        preprocess_data(test)
        epochs = 20
        LR = 0.05

    train = CustomDataset(train)
    valid = CustomDataset(valid)
    test = CustomDataset(test)


    batch_size = 1  # Set your batch size
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)
    # insert code to train simple FFNN and produce evaluation metrics
    
    feedForwardNN = FFNN(3, 2, 3)
    #print(feedForwardNN)
    
    optimizer = torch.optim.SGD(feedForwardNN.parameters(), lr=LR)
    loss = nn.CrossEntropyLoss()
    train_loss = []
    validation_loss = []
    for t in range(epochs):
        #print(f"Epoch {t+1}\n------------------------------- \n")
        train_loss.append(trainModel(train_loader, feedForwardNN, loss, optimizer, device))
        testData = testModel(valid_loader, feedForwardNN, loss)
        validation_loss.append(testData[0])

        if(early_stopping and testData[1] >= 84):
            break

    # Could add a condition that interrupts training when the loss doesn't change much
    #print('Done!')

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_loss)+1), train_loss, label='Training Loss')
    plt.plot(range(1, len(validation_loss)+1), validation_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss Over Epochs')
    plt.legend()
    plt.show()
    plt.show()

    return evaluate_model(test_loader, feedForwardNN, loss)


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet,self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128,64)  # Flatten the image and then apply linear transformation
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the image
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return softmax(x)   

def read_mnist(file_name):
    
    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label,attribs])
    return(data_set)
        
def processDataMNIST(dataset):
    labels = []
    features = []

    for i in range(len(dataset)):
        label = int(dataset[i][0])
        grayscale_values = [int(x) for x in dataset[i][1]]
        
        # Map grayscale values to binary using a threshold (adjust threshold as needed)
        threshold = 128  # Example threshold: Change this to suit your needs
        binary_values = [1 if x > threshold else 0 for x in grayscale_values]
        
        features.append(binary_values)
        labels.append(label)

    return features, labels
        
def show_mnist(file_name,mode):
    
    data_set = read_mnist(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ',end='')
                else:
                    print('*',end='')
            else:
                print('%4s ' % data_set[obs][1][idx],end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0],end='')
        print(' ')

def classify_mnist(device):
    
    train_data = read_mnist('mnist_train.csv')
    valid_data = read_mnist('mnist_valid.csv')
    test_data = read_mnist('mnist_test.csv')

    train_features, train_labels = processDataMNIST(train_data)
    valid_features, valid_labels = processDataMNIST(valid_data)
    test_features, test_labels = processDataMNIST(test_data)

    train = CustomMnistDataset(list(zip(train_labels, train_features)))
    valid = CustomMnistDataset(list(zip(valid_labels, valid_features)))
    test = CustomMnistDataset(list(zip(test_labels, test_features)))

    batch_size = 64  # Set your batch size
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)

    # ATTENTION :: CONVERT THE DATA TO INT BEFORE USING IT
    #show_mnist('mnist_test.csv','pixels')
    
    mnistNetModel = MnistNet()

    optimizer = torch.optim.Adam(mnistNetModel.parameters(), lr=0.0003)
    loss = nn.CrossEntropyLoss()
    epochs = 20
    train_loss = []
    validation_loss = []
    for t in range(epochs):
        #print(f"Epoch {t+1}\n------------------------------- \n")
        train_loss.append(trainModel(train_loader, mnistNetModel, loss, optimizer, device))
        validation_loss.append(testModel(valid_loader, mnistNetModel, loss)[0])

    # Could add a condition that interrupts training when the loss doesn't change much
    #print('Done!')

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs+1), train_loss, label='Training Loss')
    plt.plot(range(1, epochs+1), validation_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss Over Epochs')
    plt.legend()
    plt.show()
    plt.show()

    return evaluate_model(test_loader, mnistNetModel, loss)
    
def classify_mnist_reg(device):
    train = read_mnist('mnist_train.csv')
    valid = read_mnist('mnist_valid.csv')
    test = read_mnist('mnist_test.csv')

    train = CustomMnistDataset(train)
    valid = CustomMnistDataset(valid)
    test = CustomMnistDataset(test)

    batch_size = 64
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)
    
    mnistNetModel = MnistNet()

    # Adding L2 Regularization using weight_decay
    optimizer = torch.optim.Adam(mnistNetModel.parameters(), lr=0.0003, weight_decay=0.01)
    loss = nn.CrossEntropyLoss()
    epochs = 20
    train_loss = []
    validation_loss = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n------------------------------- \n")
        train_loss.append(trainModel(train_loader, mnistNetModel, loss, optimizer, device))
        validation_loss.append(testModel(valid_loader, mnistNetModel, loss)[0])

    # Could add a condition that interrupts training when the loss doesn't change much# Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs+1), train_loss, label='Training Loss')
    plt.plot(range(1, epochs+1), validation_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss Over Epochs')
    plt.legend()
    plt.show()
    plt.show()

    evaluate_model(test_loader, mnistNetModel, loss)

def classify_insurability_manual(device):
    
    train = read_insurability('three_train.csv')
    valid = read_insurability('three_valid.csv')
    test = read_insurability('three_test.csv')
    
    # reimplement classify_insurability() without using a PyTorch optimizer.
    # this part may be simpler without using a class for the FFNN
    
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    testloss , testAccuracy = classify_insurability(device,preprocess=True, early_stopping=True)
    print("For Insurable Data \n Test Loss: ", testloss, "\n Test Accuracy: ", testAccuracy, "\n Preprocessing: True \n Early Stopping: True \n")

    testloss, testAccuracy = classify_insurability(device,preprocess=False, early_stopping=True)
    print("For Insurable Data \n Test Loss: ", testloss, "\n Test Accuracy: ", testAccuracy, "\n Preprocessing: False \n Early Stopping: True \n")

    testloss, testAccuracy = classify_insurability(device,preprocess=True, early_stopping=False)
    print("For Insurable Data \n Test Loss: ", testloss, "\n Test Accuracy: ", testAccuracy, "\n Preprocessing: True \n Early Stopping: False \n")

    testloss, testAccuracy = classify_insurability(device,preprocess=False, early_stopping=False)
    print("For Insurable Data \n Test Loss: ", testloss, "\n Test Accuracy: ", testAccuracy, "\n Preprocessing: False \n Early Stopping: False \n")

    test_loss, acc = classify_mnist(device)
    print("For Mnist Data \n Test Loss: ", test_loss, "\n Test Accuracy: ", acc)
    
    classify_mnist_reg(device)
    classify_insurability_manual(device)
    
if __name__ == "__main__":
    main()
