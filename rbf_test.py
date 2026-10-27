# rbf_test.py
import numpy as np
import matplotlib.pyplot as plt

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
        self.w -= lr * delta * phi
        self.b -= lr * delta
        # Исправлено: возвращаем два значения, как forward / 修复：返回两个值，匹配拆包
        return y_pred, phi


if __name__ == "__main__":
    # ---------- Параметры RBF слоя / RBF层参数 ----------
    # Ручное задание центров / 手动给定中心点
    centers = np.array([
        [0.1],
        [0.4],
        [0.7],
        [0.9]
    ])
    sigma = 0.15   # ширина RBF / 高斯宽度σ

    # Создаём RBF-сеть / 实例化RBF网络
    net = RBFNet(centers=centers, sigma=sigma, output_dim=1)

    # Обучающая выборка / 训练样本
    x_train = np.array([0.2, 0.5, 0.8])
    y_train = np.array([0.3, 0.7, 0.2])

    epoch_num = 2000
    loss_history = []

    print("===== До обучения / 训练之前 =====")
    for xi, yi in zip(x_train, y_train):
        pred, _ = net.forward(xi)
        print(f"x={xi}, target={yi}, pred={pred:.4f}")

    for i in range(epoch_num):
        total_loss = 0
        for xi, yi in zip(x_train, y_train):
            yp, _ = net.train_step(xi, yi, lr=0.2)
            total_loss += (yp - yi) ** 2
        loss_history.append(total_loss / len(x_train))

    print("\n===== После обучения / 训练之后 =====")
    for xi, yi in zip(x_train, y_train):
        pred, _ = net.forward(xi)
        print(f"x={xi}, target={yi}, pred={pred:.4f}")

    # График ошибки / 绘制损失曲线
    plt.figure(figsize=(8,4))
    plt.plot(loss_history)
    plt.xlabel("Эпоха / Epoch")
    plt.ylabel("MSE Ошибка / MSE Loss")
    plt.title("Обучение RBF сети (RBF слой фиксирован) / RBF Network Training Curve")
    plt.grid(True)
    plt.savefig("rbf_loss.png", dpi=300)
    plt.show()
    print("\n✅ График сохранён как rbf_loss.png / 图片已保存 rbf_loss.png")

