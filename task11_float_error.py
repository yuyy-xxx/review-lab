import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

iris = load_iris()
X, y = iris.data, iris.target
model = LogisticRegression(max_iter=200)
model.fit(X, y)

initial_type = [("float_input", FloatTensorType([None, 4]))]
onx = convert_sklearn(model, initial_types=initial_type)
with open("logreg_float.onnx", "wb") as f:
    f.write(onx.SerializeToString())

X_test = X[:1]
print("Тип данных X_test (до исправления):", X_test.dtype)

sess = ort.InferenceSession("logreg_float.onnx")
input_name = sess.get_inputs()[0].name

print("\n--- Попытка запуска с float64 (должна быть ошибка) ---")
try:
    pred_bad = sess.run(None, {input_name: X_test})
    print("Успех (не должно так быть!)")
except Exception as e:
    print("Ошибка:")
    print(e)

print("\n--- Исправление: преобразуем в float32 ---")
X_test_f32 = X_test.astype(np.float32)
print("Тип данных X_test (после исправления):", X_test_f32.dtype)
pred_ok = sess.run(None, {input_name: X_test_f32})
print("Результат предсказания:", pred_ok)
print("✅ Запуск с float32 работает!")
