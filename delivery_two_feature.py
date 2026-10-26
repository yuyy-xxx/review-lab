import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression

np.random.seed(42)

# 幻灯片固定数据集
X = np.array([
    [1, 0.5], [3, 1.0], [5, 2.0], [7, 3.0], [9, 4.0],
    [2, 1.5], [4, 2.5], [6, 3.5], [8, 4.5], [10, 5.0]
], dtype=np.float32)
# 公式 цена = 100 + 50 * расстояние + 30 * вес
y = (100 + 50 * X[:, 0] + 30 * X[:, 1]).astype(np.float32)

# ---- sklearn ----
model_sk = LinearRegression()
model_sk.fit(X, y)
pred_sk = model_sk.predict([[5, 2]])[0]

# ---- PyTorch ----
X_t = torch.tensor(X)
y_t = torch.tensor(y.reshape(-1, 1))
model_pt = nn.Linear(2, 1)
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model_pt.parameters(), lr=0.0001)

for _ in range(2000):
    y_pred = model_pt(X_t)
    loss = loss_fn(y_pred, y_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ========== 和这页幻灯片一模一样的输出代码 ==========
with torch.no_grad():
    pred_pt = model_pt(torch.tensor([[5.0, 2.0]])).item()

print(f"Для заказа 5 км, 2 кг:")
print(f"Истинное значение: {100 + 50*5 + 30*2}") # 410
print(f"sklearn: {pred_sk:.2f}")
print(f"PyTorch: {pred_pt:.2f}")
