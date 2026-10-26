import numpy as np

# 激活函数
def relu(z):
    return np.maximum(0, z)

def tansig(z):
    return np.tanh(z)

def logsig(z):
    return 1.0 / (1.0 + np.exp(-z))

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

# 激活函数字典
ACTIVATIONS = {
    "linear": lambda z: z,
    "relu": relu,
    "tansig": tansig,
    "logsig": logsig,
    "softmax": softmax
}

class Neuron:
    def __init__(self, n_inputs, activation="linear"):
        if activation not in ACTIVATIONS:
            raise ValueError("不支持该激活函数")
        self.activation_name = activation
        self.activation_func = ACTIVATIONS[activation]
        # 随机初始化权重、偏置
        self.weights = np.random.randn(n_inputs) * 0.1
        self.bias = 0.1

    def forward(self, x):
        z = np.dot(self.weights, x) + self.bias
        return self.activation_func(z)


if __name__ == "__main__":
    x = np.array([1.0, 2.0, 3.0])
    for name in ["linear", "relu", "tansig", "logsig", "softmax"]:
        n = Neuron(n_inputs=3, activation=name)
        out = n.forward(x)
        print(f"{name:6s} -> {out}")
