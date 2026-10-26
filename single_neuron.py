import numpy as np

# 激活函数
def linear(z):
    return z

def relu(z):
    return np.maximum(0, z)

def tan_sig(z):
    return np.tanh(z)

# 激活函数字典，函数式映射选择
activation_map = {
    "linear": linear,
    "relu": relu,
    "tansig": tan_sig
}

# 单个神经元实现 y = f( sum(x_i * w_i) + b )
def neuron(x, w, b, act_name="relu"):
    z = np.dot(x, w) + b
    f = activation_map[act_name]
    return f(z)

# ========== 测试示例 ==========
if __name__ == "__main__":
    x = np.array([1.0, 2.0, 3.0])
    w = np.array([0.2, 0.5, -0.1])
    b = 0.3

    y1 = neuron(x, w, b, act_name="linear")
    y2 = neuron(x, w, b, act_name="relu")
    y3 = neuron(x, w, b, act_name="tansig")

    print("-" * 30)
    print(f"linear 输出: {y1:.4f}")
    print(f"relu   输出: {y2:.4f}")
    print(f"tansig 输出: {y3:.4f}")
