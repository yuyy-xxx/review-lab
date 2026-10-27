# RBFNet.py
import numpy as np

class RBFLayer:
    def __init__(self, centers, sigma):
        """
        centers: матрица центров RBF нейронов / RBF中心点矩阵，shape = [n_rbf, input_dim]
        sigma: ширина функции RBF / RBF高斯宽度σ，固定，不训练
        """
        self.centers = np.array(centers, dtype=np.float64)
        self.sigma = sigma

    def rbf_gauss(self, x, c):
        # Гауссова радиально-базисная функция / 高斯径向基函数
        dist_sq = np.sum((x - c) ** 2)
        return np.exp(- dist_sq / (2 * self.sigma ** 2))

    def forward(self, x):
        # Прямой проход RBF слоя (слой не обучается) / RBF层前向传播，本层不训练
        out = np.array([self.rbf_gauss(x, c) for c in self.centers])
        return out


class RBFNet:
    def __init__(self, centers, sigma, output_dim=1):
        """
        RBF сеть: RBFLayer (скрытый слой, не обучается) + линейный выходной слой (обучается)
        RBF网络结构：RBF隐藏层（固定不训练） + 线性输出层（可训练）
        centers: центры RBF / RBF中心点
        sigma: ширина RBF / 高斯宽度
        output_dim: размер выхода / 输出维度
        """
        self.rbf_layer = RBFLayer(centers, sigma)
        n_rbf = len(centers)
        # Линейный слой: веса + смещение / 线性层权重偏置（唯一可训练部分）
        self.w = np.random.randn(n_rbf) * 0.1
        self.b = 0.0

    def forward(self, x):
        # Прямой проход всей сети / 整个网络前向传播
        phi = self.rbf_layer.forward(x)
        y = np.dot(phi, self.w) + self.b
        return y, phi

    def train_step(self, x, target, lr=0.1):
        """
        Обучение только линейного выходного слоя / 只训练后面线性层，RBF中心点、sigma固定不变
        x: входной вектор / 输入
        target: целевое значение / 目标
        lr: скорость обучения / 学习率
        """
        y_pred, phi = self.forward(x)
        delta = y_pred - target
        # обновляем только w и b линейного слоя
        self.w -= lr * delta * phi
        self.b -= lr * delta
        return y_pred
