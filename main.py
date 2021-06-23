print("Hello World!")

import os
import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.preprocessing import LabelEncoder

##############Data pre processing
dataset = pd.read_csv("seguros_training_data.csv")
test_dataset = pd.read_csv("seguros_testing_data.csv")
#Set-up the X(training info) and y(training target) arrays.
X = dataset.iloc[:, :-1]#try.values
y = dataset.iloc[:, 10]
#Set-up the X2(test info) and y2(training target) arrays.
X2 = test_dataset.iloc[:, :-1]
y2 = test_dataset.iloc[:, 10]

X = X.apply(LabelEncoder().fit_transform)
X2 = X2.apply(LabelEncoder().fit_transform)


test_var = X.values
print("test")
print(type(test_var))
print(test_var)
tensor_datass = torch.from_numpy(test_var)
print("test2")
print(tensor_datass)






device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 2),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

data = [0, 0]
# X = test_var
X = torch.tensor(data)
X = X.View()
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")











