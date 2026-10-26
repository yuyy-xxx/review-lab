import numpy as np
from sklearn.linear_model import LinearRegression

# Данные: расстояние (км) -> стоимость (руб.)
X = np.array([[1], [3], [5], [7], [9]])
y = np.array([150, 250, 350, 450, 550])

# Обучение
model = LinearRegression()
model.fit(X, y)

# Функция-обёртка
def predict_delivery(distance):
    return model.predict([[distance]])[0]

# Проверка
for dist in [2, 6, 10]:
    print(f"Расстояние {dist} км -> доставка: {predict_delivery(dist):.2f} руб.")
