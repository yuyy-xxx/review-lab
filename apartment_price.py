from sklearn.linear_model import LinearRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import numpy as np

# 数据集
X = [[30], [50], [70], [90], [110]]
y = [3000, 5000, 7000, 9000, 11000]

# 训练模型
model = LinearRegression()
model.fit(X, y)

# 校验点：预测60㎡
test_area = np.array([[60.0]])
pred = model.predict(test_area)
print(f"Предсказанная цена для 60 м² / 60平米预测价格: {pred[0]} тыс. руб.")

# 导出ONNX
initial_type = [("area_input", FloatTensorType([None, 1]))]
onx_model = convert_sklearn(model, initial_types=initial_type)
with open("apartment_model.onnx", "wb") as f:
    f.write(onx_model.SerializeToString())

# ONNX Runtime推理
sess = ort.InferenceSession("apartment_model.onnx", providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
onnx_pred = sess.run(None, {input_name: test_area.astype(np.float32)})
print(f"Результат из ONNX Runtime / ONNX推理结果: {onnx_pred[0][0]} тыс. руб.")
