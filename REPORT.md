# Отчёт по лабораторной работе: предсказание цены квартиры
# 实验报告：房屋房价预测

## 1. Постановка задачи
## 1. 任务描述
Вход: площадь квартиры (м²) — одно число
Выход: цена (тыс. руб.) — одно число
Тип задачи: регрессия (предсказание числа, а не классификация)

输入：房屋面积（平方米）——单个数值
输出：价格（千卢布）——单个数值
任务类型：回归任务（预测数值，不是分类）

Данные / 数据：
| Площадь, м² | Цена, тыс. руб. |
| ---- | ---- |
| 30 | 3000 |
| 50 | 5000 |
| 70 | 7000 |
| 90 | 9000 |
| 110 | 11000 |

Контрольная точка: предсказать цену для 60 м²
测试点：预测面积60平方米对应的房价

## 2. Модель 1: линейная регрессия scikit-learn + экспорт ONNX
## 2. 模型1：scikit-learn 线性回归 + ONNX导出推理
```python
from sklearn.linear_model import LinearRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import numpy as np

# Данные: площадь (м²) → цена (тыс. руб.)
X = [[30], [50], [70], [90], [110]]
y = [3000, 5000, 7000, 9000, 11000]

# Обучение
model = LinearRegression()
model.fit(X, y)

# Предсказание
test_area = np.array([[60.0]])
pred = model.predict(test_area)
print(f"Прогноз для 60 м²: {pred[0]}")

# Экспорт модели в формат ONNX
initial_type = [("area_input", FloatTensorType([None, 1]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)
with open("apartment_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# Инференция через ONNX Runtime
sess = ort.InferenceSession("apartment_model.onnx", providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
onnx_pred = sess.run(None, {input_name: test_area.astype(np.float32)})
print(f"Результат из ONNX Runtime: {onnx_pred[0][0][0]}")

## 6. Классификация цветов RGB с помощью MLP
## 6. RGB颜色多层感知机分类实验
```python
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. 生成模拟RGB数据：白色类别、红色类别
np.random.seed(42)
n_samples = 200

white = np.random.normal([255, 255, 255], 10, (n_samples, 3))
red = np.random.normal([255, 0, 0], 10, (n_samples, 3))

X = np.vstack([white, red])
y = np.array([0] * n_samples + [1] * n_samples)

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 构建并训练MLP模型
mlp = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)

# 5. 在测试集上评估
y_pred = mlp.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.4f}")

# 6. 可视化测试集预测结果
plt.figure(figsize=(8, 6))
plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=y_pred,
    cmap="coolwarm",
    edgecolor="k",
    s=80
)
plt.xlabel("R channel")
plt.ylabel("G channel")
plt.title("MLP predictions on test set")
plt.colorbar(ticks=[0, 1], label="Predicted class")
plt.clim(-0.5, 1.5)
plt.show()

# 7. 定义颜色预测函数
def predict_color(model, scaler, color):
    color = np.array(color).reshape(1, -1)
    color_scaled = scaler.transform(color)
    prediction = model.predict(color_scaled)
    class_names = {0: "white", 1: "red"}
    return class_names[prediction[0]]

# 8. 预测新颜色
new_color_1 = [251, 10, 10]
new_color_2 = [250, 250, 250]

result_1 = predict_color(mlp, scaler, new_color_1)
result_2 = predict_color(mlp, scaler, new_color_2)

print(f"Color {new_color_1} is predicted as: {result_1}")
print(f"Color {new_color_2} is predicted as: {result_2}")
