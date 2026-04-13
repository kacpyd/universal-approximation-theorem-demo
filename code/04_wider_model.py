import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. Create denser data for the harder function
X = torch.linspace(-1.5, 1.5, 2000).view(-1, 1).float()
y = torch.sin(70 * X) * torch.exp(X) / 7.

# 2. Move data to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X, y = X.to(device), y.to(device)

# 3. Define the same shallow model
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

# 4. Increase model width: 20 -> 100
model = PerceptronDeep(hidden_size=100).to(device)

# 5. Keep Adam
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 6. Training
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

# 7. Plot training loss
plt.plot(train_loss)
plt.xlabel("epoch")
plt.yscale("log")
plt.show()

# 8. Plot prediction vs target
model.eval()
with torch.no_grad():
    y_hat = model(X)

plt.plot(X.cpu().numpy(), y.cpu().numpy(), 'g', label='target')
plt.plot(X.cpu().numpy(), y_hat.cpu().numpy(), label='prediction')
plt.xlabel("x")
plt.legend()
plt.show()
