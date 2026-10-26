import numpy as np

# ========== 版本1：函数式神经元 ==========
def single_neuron(x, w, b, activate="relu"):
    z = np.dot(x, w) + b
    if activate == "relu":
        return max(0, z)
    elif activate == "sigmoid":
        return 1/(1+np.exp(-z))
    elif activate == "tanh":
        return np.tanh(z)
    else:
        return z

# ========== 版本2：类神经元 ==========
class Neuron:
    def __init__(self, input_num, activate="relu"):
        self.w = np.random.randn(input_num)*0.1
        self.b = 0.0
        self.activate = activate

    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        if self.activate == "relu":
            return max(0, z)
        elif self.activate == "sigmoid":
            return 1/(1+np.exp(-z))
        elif self.activate == "tanh":
            return np.tanh(z)
        else:
            return z

# ========== 对比测试 ==========
if __name__ == "__main__":
    np.random.seed(42)
    n_in = 3
    x_rand = np.random.randn(n_in)
    w_rand = np.random.randn(n_in) * 0.1
    b_rand = np.random.randn() * 0.1
    act = "relu"

    res_func = single_neuron(x_rand, w_rand, b_rand, act)

    neuron = Neuron(n_in, act)
    neuron.w = w_rand
    neuron.b = b_rand
    res_class = neuron.forward(x_rand)

    print("随机输入 x:      ", x_rand)
    print("随机权重 w:      ", w_rand)
    print("随机偏置 b:      ", b_rand)
    print("激活函数:         ", act)
    print("函数版输出:       ", res_func)
    print("类版本输出:       ", res_class)
    print("差值（应接近0）:  ", abs(res_func - res_class))
