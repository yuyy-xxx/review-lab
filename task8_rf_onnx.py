import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

# 1. Данные
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# 2. Обучение RandomForest
clf = RandomForestClassifier(n_estimators=10, random_state=42)
clf.fit(X_train, y_train)

# 3. Экспорт в ONNX
initial_type = [('float_input', FloatTensorType([None, 4]))]
onx = convert_sklearn(clf, initial_types=initial_type)
with open("rf_iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())

# 4. Запуск через ONNX Runtime
sess = ort.InferenceSession("rf_iris.onnx")
input_name = sess.get_inputs()[0].name
pred_onnx = sess.run(None, {input_name: X_test.astype(np.float32)})[0]

# 5. Сравнение
pred_sklearn = clf.predict(X_test)
print(f"Совпадение: {(pred_onnx == pred_sklearn).mean() * 100:.2f}%")
print(f"Предсказания ONNX (первые 5): {pred_onnx[:5]}")
print(f"Предсказания sklearn (первые 5): {pred_sklearn[:5]}")

