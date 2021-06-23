print("Hello World!")

import os
import torch
import torch.nn as nn
import torch.nn.functional as n
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.preprocessing import LabelEncoder

# ##############Data pre processing
# dataset = pd.read_csv("seguros_training_data.csv")
# test_dataset = pd.read_csv("seguros_testing_data.csv")
# #Set-up the X(training info) and y(training target) arrays.
# X = dataset.iloc[:, :-1]#try.values
# y = dataset.iloc[:, 10]
# #Set-up the X2(test info) and y2(training target) arrays.
# X2 = test_dataset.iloc[:, :-1]
# y2 = test_dataset.iloc[:, 10]
#
# X = X.apply(LabelEncoder().fit_transform)
# X2 = X2.apply(LabelEncoder().fit_transform)


# test_var = X.values
# print("test")
# print(type(test_var))
# print(test_var)
# tensor_datass = torch.from_numpy(test_var)
# print("test2")
# print(tensor_datass)






device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cuda:0'
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


optimizer = optim.Adam(model.parameters(), lr=0.001)

corridas = 3

for i in range(corridas):
    data = [float(0), float(0)]
    X = torch.tensor(data, device=device)
    X = X.view(-1, 2)
    data = [0, 0]
    # data = [float(0), float(0)] #new
    y = torch.tensor(data, device=device)
    y = y.view(-1, 2)
    model.zero_grad()
    logits = model(X)
    #pred_probab = nn.Softmax(dim=1)(logits)
    pred_probab = n.log_softmax(logits, dim=1) #new
    print(pred_probab)
    print("Stop")
    print(y)
    loss = n.nll_loss(pred_probab,target=y)
    loss.backwards()
    optimizer.step()



#print(X.shape)


#y_pred = pred_probab.argmax(1)
#print(f"Predicted class: {y_pred}")








#Trrash
# loss = nn.NLLLoss(pred_probab,y) #New


