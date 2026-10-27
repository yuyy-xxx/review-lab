import os
import pickle
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

#1. Обучение
iris = load_iris()
clr = LogisticRegression(max_iter=200)
clr.fit(iris.data, iris.target)

#2. Сохранение через pickle
with open("logreg_iris.pkl", "wb") as f:
    pickle.dump(clr, f)

#3. Экспорт в ONNX
initial_type = [("float_input", FloatTensorType([None,4]))]
onx = convert_sklearn(clr, initial_types=initial_type)
with open("logreg_iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())

#4. Сравнение размеров
pkl_size = os.path.getsize("logreg_iris.pkl")
onnx_size = os.path.getsize("logreg_iris.onnx")

print(f"Размер .pkl: {pkl_size / 1024:.2f} КБ")
print(f"Размер .onnx: {onnx_size / 1024:.2f} КБ")
print(f"Отношение: {onnx_size / pkl_size:.2f}")

