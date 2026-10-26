import torch
import torch.nn as nn
import torch.optim as optim

# Данные
X = torch.tensor([[1.0], [3.0], [5.0], [7.0], [9.0]])
y = torch.tensor([[150.0], [250.0], [350.0], [450.0], [550.0]])

# Модель
model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001)

# Обучение с логированием
for epoch in range(1000):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d}: loss = {loss.item():.2f}")

# Предсказание
with torch.no_grad():
    print(f"\nПредсказание для 6 км: {model(torch.tensor([[6.0]])).item():.2f} руб.")

