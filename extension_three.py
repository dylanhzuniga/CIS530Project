import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from features import readInDataFeatureVector

NUM_EPOCHS = 40
LEARNING_RATE = 0.001

class Dataset(Dataset):
  def __init__(self, X, y):
    y = y.astype(int)
    self.X = torch.from_numpy(X).float()
    self.y = torch.from_numpy(y).long()
  
  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]

class FNN(nn.Module):
  def __init__(self, input_size=18, hidden_size=5000, output_size=2):
    super(FNN, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    out = F.relu(self.fc1(x))
    out = self.fc2(out)
    return out

def train(model, loader, criterion, optimizer):
  for _ in range(NUM_EPOCHS):
    for X, y in loader:
      model.zero_grad()
      output = model(X)
      loss = criterion(output, y.view(X.shape[0]))
      loss.backward()
      optimizer.step()

def predict(model, loader):
  predicted_labels = []
  for X, _ in loader:
    output = model(X)
    _, predicted = torch.max(output.data, 1)
    predicted_labels.extend(predicted)
  return [x.item() for x in predicted_labels]

if __name__ == "__main__":
  trainfilename = "./data/train.data"
  devfilename = "./data/dev.data"
  testfilename  = "./data/test.data"
  outputfilename = "./outputs/extensionthree.output"

  train_X, train_y = readInDataFeatureVector(trainfilename)
  train_dataset = Dataset(train_X, train_y.reshape(-1, 1))
  train_loader = DataLoader(train_dataset)

  test_X, test_y = readInDataFeatureVector(testfilename)
  test_dataset = Dataset(test_X, test_X)
  test_loader = DataLoader(test_dataset, shuffle=False)

  model = FNN()
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adagrad(model.parameters(), lr=LEARNING_RATE)

  train(model, train_loader, criterion, optimizer)

  predictions = predict(model, test_loader)

  with open(outputfilename, 'w', encoding='utf8') as f:
    prob = 0
    for p in predictions:
      if p == 1:
        f.write("true\t" + "{0:.4f}".format(prob) + "\n")
      else:
        f.write("false\t" + "{0:.4f}".format(prob) + "\n")
    f.close()
