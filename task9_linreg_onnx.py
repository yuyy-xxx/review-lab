import numpy as np
from sklearn.linear_model import LinearRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

# 1. Данные из лекции 1
X = np.array([[30], [50], [70], [90], [110]], dtype=np.float32)
y = np.array([3000, 5000, 7000, 9000, 11000], dtype=np.float32)

# 2. Обучение
model = LinearRegression()
model.fit(X, y)

# 3. Экспорт (вход: 1 признак)
initial_type = [("float_input", FloatTensorType([None, 1]))]
onx = convert_sklearn(model, initial_types=initial_type)
with open("linreg_price.onnx", "wb") as f:
    f.write(onx.SerializeToString())

# 4. Запуск через ONNX Runtime
sess = ort.InferenceSession("linreg_price.onnx")
input_name = sess.get_inputs()[0].name
test_input = np.array([[60]], dtype=np.float32)
pred_onnx = sess.run(None, {input_name: test_input})[0]

# 5. Сравнение
pred_sklearn = model.predict([[60]])
print(f"sklearn: {pred_sklearn[0]:.2f}")
print(f"ONNX:    {pred_onnx[0][0]:.2f}")
print(f"Разница: {abs(pred_sklearn[0] - pred_onnx[0][0]):.6f}")

