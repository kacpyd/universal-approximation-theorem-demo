import torch
import matplotlib.pyplot as plt

# denser sampling
X = torch.linspace(-1.5, 1.5, 2000).view(-1, 1).float()
y = torch.sin(70 * X) * torch.exp(X) / 7.

plt.plot(X.numpy(), y.numpy(), 'g')
plt.xlabel('x')
plt.show()
