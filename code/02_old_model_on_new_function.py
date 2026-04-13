# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 00:38:16 2026

@author: kacpe
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# STEP 1: create data first
X = torch.linspace(-1.5, 1.5, 2000).view(-1, 1).float()
y = torch.sin(70 * X) * torch.exp(X) / 7.

# STEP 2: then move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X, y = X.to(device), y.to(device)

class PerceptronDeep(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_layer = nn.Linear(1, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x

model = PerceptronDeep(hidden_size=20).to(device)

optimizer = optim.SGD(model.parameters(), lr=0.2)
criterion = nn.MSELoss()

N_epochs = 50000
train_loss = []

model.train()
for _ in range(N_epochs):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    train_loss.append(loss.item())

plt.plot(train_loss)
plt.xlabel("epoch")
plt.yscale("log")
plt.show()

model.eval()
with torch.no_grad():
    y_hat = model(X)

plt.plot(X.cpu().numpy(), y.cpu().numpy(), 'g', label='target')
plt.plot(X.cpu().numpy(), y_hat.cpu().numpy(), label='prediction')
plt.xlabel("x")
plt.legend()
plt.show()