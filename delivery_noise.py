import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

torch.manual_seed(42)

X = np.array([[1.0], [3.0], [5.0], [7.0], [9.0]])
y_true = np.array([[150.0], [250.0], [350.0], [450.0], [550.0]])

# 添加噪声 N(0,30)
noise = np.random.normal(loc=0, scale=30, size=y_true.shape)
y_noisy = y_true + noise

# 1. sklearn
model_sk = LinearRegression()
model_sk.fit(X, y_noisy)
pred_sk = model_sk.predict([[6]])[0][0]

# 2. PyTorch
X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y_noisy, dtype=torch.float32)
model_pt = nn.Linear(1,1)
loss_fn = nn.MSELoss()
opt = optim.SGD(model_pt.parameters(), lr=0.0001)

for _ in range(1000):
    y_pred = model_pt(X_t)
    loss = loss_fn(y_pred, y_t)
    opt.zero_grad()
    loss.backward()
    opt.step()

with torch.no_grad():
    pred_pt = model_pt(torch.tensor([[6.0]])).item()

# ========== 和幻灯片一模一样的打印输出 ==========
print("Истинное значение для 6 км: 400")
print(f"sklearn: {pred_sk:.2f}")
print(f"PyTorch: {pred_pt:.2f}")

# ========== 绘图 ==========
plt.scatter(X.flatten(), y_noisy, label='Данные с шумом')
plt.plot(X.flatten(), model_sk.predict(X), 'r-', label='sklearn')
plt.legend()
plt.title("Зависимость цены доставки от расстояния")
plt.xlabel("Расстояние, км")
plt.ylabel("Цена доставки, руб.")
plt.savefig("noise_plot.png", dpi=300)
# plt.show()
