from sklearn.linear_model import LinearRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ========== 1. scikit-learn 线性回归 + ONNX导出 ==========
print("===== scikit-learn 线性回归 =====")
X = [[30], [50], [70], [90], [110]]
y = [3000, 5000, 7000, 9000, 11000]

# 训练模型
model = LinearRegression()
model.fit(X, y)

# 校验点：预测60㎡
test_area = np.array([[60.0]])
pred = model.predict(test_area)
print(f"Прогноз для 60 м²: {pred[0]}")

# 导出ONNX
initial_type = [("area_input", FloatTensorType([None, 1]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)
with open("apartment_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# ONNX Runtime推理
sess = ort.InferenceSession("apartment_model.onnx", providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
onnx_pred = sess.run(None, {input_name: test_area.astype(np.float32)})
print(f"Результат из ONNX Runtime: {onnx_pred[0][0][0]}")

# ========== 2. PyTorch 线性回归（带归一化） ==========
print("\n===== PyTorch 线性回归 =====")
X_t = torch.tensor([[30.0], [50.0], [70.0], [90.0], [110.0]])
y_t = torch.tensor([[3000.0], [5000.0], [7000.0], [9000.0], [11000.0]])

# 特征归一化
x_mean = X_t.mean()
x_std = X_t.std()
X_norm = (X_t - x_mean) / x_std

model_t = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model_t.parameters(), lr=0.1)

epochs = 3000
for epoch in range(epochs):
    y_pred = model_t(X_norm)
    loss = loss_fn(y_pred, y_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 预测60㎡
x_test = torch.tensor([[60.0]])
x_test_norm = (x_test - x_mean) / x_std
with torch.no_grad():
    res = model_t(x_test_norm)
print("Прогноз для 60 м²:", res.item())
