import random, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ========= 激活函数和导数 =========
def sigmoid(z):
    return 1.0/(1.0 + math.exp(-z))
def sigmoid_der(z):
    s = sigmoid(z)
    return s*(1-s)

def tanh_act(z):
    return math.tanh(z)
def tanh_der(z):
    t = math.tanh(z)
    return 1 - t*t

def relu(z):
    return max(0.0, z)
def relu_der(z):
    return 1.0 if z>0 else 0.0

# ========= 训练函数（向量 wxb=[w,b]） =========
def train_act(dataset, act_func, act_der, lr=0.01, epoch=5000):
    wxb = [0.1, 0.1]
    for e in range(epoch):
        grad_sum = [0.0,0.0]
        for x, y_true in dataset:
            wx, wb = wxb
            z = wx * x + wb
            y_pred = act_func(z)
            error = y_pred - y_true
            der_total = error * act_der(z)
            xn = [x, 1.0]
            grad = [der_total * xn[0], der_total * xn[1]]
            grad_sum[0] += grad[0]
            grad_sum[1] += grad[1]
        grad_avg = [grad_sum[0]/len(dataset), grad_sum[1]/len(dataset)]
        wxb[0] -= lr * grad_avg[0]
        wxb[1] -= lr * grad_avg[1]
    return wxb

# ===== 1. Sigmoid 数据集：标签 0 / 1（二分类） =====
ds_sigmoid = [[1.0, 0.05], [2.0, 0.12], [4.0, 0.88], [5.0, 0.94]]
w_sig,b_sig = train_act(ds_sigmoid, sigmoid, sigmoid_der, lr=0.08)
print(f"Sigmoid result: w={w_sig:.2f}, b={b_sig:.2f}")

# ===== 2. Tanh 数据集：标签 -1 ~ +1 =====
ds_tanh = [[1.0, -0.8], [2.0, -0.3], [4.0, 0.4], [5.0, 0.85]]
w_tan,b_tan = train_act(ds_tanh, tanh_act, tanh_der, lr=0.05)
print(f"Tanh result: w={w_tan:.2f}, b={b_tan:.2f}")

# ===== 3. ReLU 数据集：标签 >=0 =====
ds_relu = [[1.0, 2.2], [2.0, 4.1], [4.0, 8.3], [5.0, 10.4]]
w_rel,b_rel = train_act(ds_relu, relu, relu_der, lr=0.005)
print(f"ReLU result: w={w_rel:.2f}, b={b_rel:.2f}")

# ========= 绘图 charts =========
fig, axes = plt.subplots(1,3, figsize=(15,4))
x_line = [i/10 for i in range(0,60)]

# Sigmoid
ax=axes[0]
xd = [p[0] for p in ds_sigmoid]
yd = [p[1] for p in ds_sigmoid]
yp_sig = [sigmoid(w_sig*xi + b_sig) for xi in x_line]
ax.scatter(xd,yd,c='green', label='Data')
ax.plot(x_line, yp_sig, 'r-', label=f'y=sigmoid({w_sig:.2f}x+{b_sig:.2f})')
ax.set_title("Sigmoid (0~1)")
ax.legend()
ax.grid(alpha=0.3)

# Tanh
ax=axes[1]
xd = [p[0] for p in ds_tanh]
yd = [p[1] for p in ds_tanh]
yp_tan = [tanh_act(w_tan*xi + b_tan) for xi in x_line]
ax.scatter(xd,yd,c='green', label='Data')
ax.plot(x_line, yp_tan, 'r-', label=f'y=tanh({w_tan:.2f}x+{b_tan:.2f})')
ax.set_title("Tanh (-1~1)")
ax.legend()
ax.grid(alpha=0.3)

# ReLU
ax=axes[2]
xd = [p[0] for p in ds_relu]
yd = [p[1] for p in ds_relu]
yp_rel = [relu(w_rel*xi + b_rel) for xi in x_line]
ax.scatter(xd,yd,c='green', label='Data')
ax.plot(x_line, yp_rel, 'r-', label=f'y=ReLU({w_rel:.2f}x+{b_rel:.2f})')
ax.set_title("ReLU (>=0)")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("activation_chart.png", dpi=300)
plt.close()
print("✅ 图表已经保存为 activation_chart.png")
