import matplotlib.pyplot as plt

import torch as torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

#%%

# 1. Générer les données (fonction à apprendre : y = x^2)
x = torch.linspace(-2, 2, 400).unsqueeze(1)
y = torch.exp(torch.sin(x**2))

# 2. Définir le modèle
model = nn.Sequential(
    nn.Linear(1, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

# 3. Définir la loss et l'optimiseur
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Entraînement
for epoch in range(1500):
    pred = model(x)                  # forward
    loss = criterion(pred, y)        # calcul erreur

    optimizer.zero_grad()            # reset gradients
    loss.backward()                  # backpropagation
    optimizer.step()                 # mise à jour

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# 5. Visualisation
x_np = x.detach().cpu().numpy().squeeze()
y_np = y.detach().cpu().numpy().squeeze()
pred_np = model(x).detach().cpu().numpy().squeeze()

plt.scatter(x_np, y_np, label="vrai")
plt.scatter(x_np, pred_np, label="prédit")
plt.legend()
plt.show()

#%%


# data
x = torch.linspace(-2, 2, 100).unsqueeze(1)
y = torch.sin(x**2)

# model
model = nn.Sequential(
    nn.Linear(1, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

fig, ax = plt.subplots()

line_pred, = ax.plot([], [], label="pred")
line_true, = ax.plot(x.numpy(), y.numpy(), label="true")
ax.legend()

def update(frame):
    global x, y, model, optimizer

    # training step
    pred = model(x)
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # update plot
    with torch.no_grad():
        y_pred = model(x)

    line_pred.set_data(x.numpy(), y_pred.numpy())

    return line_pred,

anim = FuncAnimation(
    fig,
    update,
    frames=50,
    interval=30,   # affichage fluide
    blit=False
)
anim.save(
    "C:/Users/acer/Desktop/animation.mp4",
    writer="ffmpeg",
    fps=10,
    dpi=150
)
