from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
import numpy as np
import onnxruntime as ort
from sklearn.linear_model import LinearRegression

# Обучаем модель / 训练模型
X_train = np.random.rand(20,4).astype(np.float32)
y_train = np.random.rand(20,1).astype(np.float32)
model = LinearRegression()
model.fit(X_train, y_train)

# 1. Модель: динамический batch [None,4] / 动态批次模型
initial_type_dyn = [("X", FloatTensorType([None, 4]))]
onnx_dyn = to_onnx(model, X_train[:1], initial_types=initial_type_dyn)
with open("model_dynamic.onnx", "wb") as f:
    f.write(onnx_dyn.SerializeToString())

sess_dyn = ort.InferenceSession("model_dynamic.onnx")
input_name_dyn = sess_dyn.get_inputs()[0].name

print("===== Модель с None (динамический batch) =====")
x_batch1 = np.random.rand(1,4).astype(np.float32)
res1 = sess_dyn.run(None, {input_name_dyn: x_batch1})
print(f"batch=1, форма вывода: {res1[0].shape}")

x_batch10 = np.random.rand(10,4).astype(np.float32)
res10 = sess_dyn.run(None, {input_name_dyn: x_batch10})
print(f"batch=10, форма вывода: {res10[0].shape}")

# 2. Модель: фиксированный batch [1,4] / 固定批次模型
initial_type_fix = [("X", FloatTensorType([1, 4]))]
onnx_fix = to_onnx(model, X_train[:1], initial_types=initial_type_fix)
with open("model_fixed.onnx", "wb") as f:
    f.write(onnx_fix.SerializeToString())

sess_fix = ort.InferenceSession("model_fixed.onnx")
input_name_fix = sess_fix.get_inputs()[0].name

print("\n===== Модель с фиксированным batch [1,4] =====")
x_fix_1 = np.random.rand(1,4).astype(np.float32)
res_fix_1 = sess_fix.run(None, {input_name_fix: x_fix_1})
print(f"batch=1, форма вывода: {res_fix_1[0].shape}")

print("Пытаемся запустить batch=10 (должна быть ошибка) / 尝试输入batch=10（预期报错）:")
try:
    res_fix_10 = sess_fix.run(None, {input_name_fix: x_batch10})
    print(f"batch=10, форма вывода: {res_fix_10[0].shape}")
except Exception as err:
    print(f"ОШИБКА / 报错：{err}")
