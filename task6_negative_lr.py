import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([[1.0], [3.0], [5.0], [7.0], [9.0]])
y = torch.tensor([[150.0], [250.0], [350.0], [450.0], [550.0]])

model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()

# 模拟 lr = -0.0001，手动更新，避开pytorch负lr校验
lr = 0.0001
losses = []

for epoch in range(100):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)

    loss.backward()
    # 等价负学习率：w = w + lr * grad
    with torch.no_grad():
        for p in model.parameters():
            p += lr * p.grad
    model.zero_grad()

    losses.append(loss.item())

print(f"Loss в начале: {losses[0]:.2f}")
print(f"Loss в конце: {losses[-1]:.2f}")
print("Loss растёт? ", losses[-1] > losses[0])
