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
