import sys
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import datetime
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, f1_score

def softmax(x):
    exp_x = torch.exp(x - torch.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum(dim=-1, keepdim=True)
class FFNN(nn.Module):
        
        def __init__(self, input_size, hidden_size, output_size):
            super(FFNN, self).__init__()
            torch.set_default_dtype(torch.float64)
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
        

def trainModel(dataset, model, loss_func, optimizer, device):
  model.train()
  train_loss = []
  batch = 0
  now = datetime.datetime.now()

  for (y,X) in dataset:
    batch +=1
    X = torch.Tensor(list(X))
    y = torch.Tensor(list(y))
    # make some predictions and get the error
    pred = model(X)
    loss = loss_func(pred.reshape(1,-1), y.type(torch.long))

    # where the magic happens
    # backpropogation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch % 100 == 0:
      loss, current = loss.item(), batch * len(X)
      iters = 10 * len(X)
      then = datetime.datetime.now()
      iters /= (then - now).total_seconds()
      print(f"loss: {loss:>6f} [{current:>5d}/{17000}] ({iters:.1f} its/sec)")
      now = then
      train_loss.append(loss)

  return train_loss


def testModel(dataset, model, loss_func):
  num_batches = 0
  model.eval()
  test_loss = 0

  with torch.no_grad():
    for y,X in dataset:
      X, y = torch.Tensor(list(X)), torch.Tensor(list(y))
      pred = model(X)
      test_loss += loss_func(pred.reshape(1,-1), y.type(torch.long))
      num_batches = num_batches + 1
  test_loss /= num_batches
  print(f"Avg Loss: {test_loss:>8f}\n")
  return test_loss

def evaluate_model(testDataset, model):
    # Ensure the model is in evaluation mode
    model.eval()

    # Lists to store true labels and predictions
    all_preds = []
    all_true = []

    with torch.no_grad():
        for target, data in testDataset:
            # Move data to the correct device
            data, target = torch.Tensor(list(data)), torch.Tensor(list(target))

            # Forward pass and get predictions
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 0)

            # Append actual and predicted values to lists
            all_true.append(target.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())

    # Calculate F1 score and confusion matrix
    f1 = f1_score(all_true, all_preds, average='weighted')
    cm = confusion_matrix(all_true, all_preds)

    return f1, cm


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
               
def classify_insurability(device):
    
    train = read_insurability('three_train.csv')
    valid = read_insurability('three_valid.csv')
    test = read_insurability('three_test.csv')
    
    # insert code to train simple FFNN and produce evaluation metrics
    
    feedForwardNN = FFNN(3, 2, 3)
    print(feedForwardNN)
    
    optimizer = torch.optim.SGD(feedForwardNN.parameters(), lr=0.0015)
    loss = nn.CrossEntropyLoss()
    epochs = 50
    train_loss = []
    test_loss = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n------------------------------- \n")
        train_loss.append(trainModel(train, feedForwardNN, loss, optimizer, device))
        test_loss.append(testModel(valid, feedForwardNN, loss))

    # Could add a condition that interrupts training when the loss doesn't change much
    print('Done!')

    plt.plot([i for i in range(len(train_loss))], torch.tensor(train_loss).mean(axis=1))
    plt.show()

    f1, cm = evaluate_model(test, feedForwardNN)
    print("F1 score: ", f1)
    print("\n ----------------------------------------- \n")
    print("Confusion matrix: \n", cm)

class IrisNet(nn.Module):
    def __init__(self,in_size,n_hidden1,n_hidden2,out_size,p=0):

        super(IrisNet,self).__init__()
        self.drop=nn.Dropout(p=p)
        self.linear1=nn.Linear(in_size,n_hidden1)
        nn.init.kaiming_uniform_(self.linear1.weight,nonlinearity='relu')
        self.linear2=nn.Linear(n_hidden1,n_hidden2)
        nn.init.kaiming_uniform_(self.linear1.weight,nonlinearity='relu')
        self.linear3=nn.Linear(n_hidden2,n_hidden2)
        nn.init.kaiming_uniform_(self.linear3.weight,nonlinearity='relu')
        self.linear4=nn.Linear(n_hidden2,out_size)
        
    def forward(self,x):
        x=F.relu(self.linear1(x))
        x=self.drop(x)
        x=F.relu(self.linear2(x))
        x=self.drop(x)
        x=F.relu(self.linear3(x))
        x=self.drop(x)
        x=self.linear4(x)
        return x
    

def classify_mnist(device):
    
    train = read_mnist('mnist_train.csv')
    valid = read_mnist('mnist_valid.csv')
    test = read_mnist('mnist_test.csv')

    # ATTENTION :: CONVERT THE DATA TO INT BEFORE USING IT
    #show_mnist('mnist_test.csv','pixels')
    
    mnistNetModel = IrisNet(784, 100, 50, 10, p=0)

    optimizer = torch.optim.Adam(mnistNetModel.parameters(), lr=0.003)
    loss = nn.CrossEntropyLoss()
    epochs = 10
    train_loss = []
    test_loss = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n------------------------------- \n")
        train_loss.append(trainModel(train, mnistNetModel, loss, optimizer, device))
        test_loss.append(testModel(valid, mnistNetModel, loss))

    # Could add a condition that interrupts training when the loss doesn't change much
    print('Done!')

    plt.plot([i for i in range(len(train_loss))], torch.tensor(train_loss).mean(axis=1))
    plt.show()

    f1, cm = evaluate_model(test, mnistNetModel)
    print("F1 score: ", f1)
    print("\n ----------------------------------------- \n")
    print("Confusion matrix: \n", cm)
    # insert code to train a neural network with an architecture of your choice
    # (a FFNN is fine) and produce evaluation metrics
    
def classify_mnist_reg(device):
    
    train = read_mnist('mnist_train.csv')
    valid = read_mnist('mnist_valid.csv')
    test = read_mnist('mnist_test.csv')
    #show_mnist('mnist_test.csv','pixels')
    
    # add a regularizer of your choice to classify_mnist()
    
def classify_insurability_manual(device):
    
    train = read_insurability('three_train.csv')
    valid = read_insurability('three_valid.csv')
    test = read_insurability('three_test.csv')
    
    # reimplement classify_insurability() without using a PyTorch optimizer.
    # this part may be simpler without using a class for the FFNN
    
    
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #classify_insurability(device)
    classify_mnist(device)
    classify_mnist_reg(device)
    classify_insurability_manual(device)
    
if __name__ == "__main__":
    main()
