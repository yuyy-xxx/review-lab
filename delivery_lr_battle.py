import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([[1.0], [3.0], [5.0], [7.0], [9.0]])
y = torch.tensor([[150.0], [250.0], [350.0], [450.0], [550.0]])

def train_with_lr(lr):
    model = nn.Linear(1, 1)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(1000):
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        pred = model(torch.tensor([[6.0]])).item()
    return lr, loss.item(), pred


for lr in [0.1, 0.0001, 0.000001]:
    lr_val, final_loss, pred = train_with_lr(lr)
    print(f"lr={lr_val:<10} loss={final_loss:<20.2f} pred(6)={pred:.2f}")
