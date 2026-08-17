import random # 随机梯度下降，随机初始化参数
import torch
from d2l import torch as d2l

# 人为生成一批符合线性回归规律的数据
def synthetic_data(w, b, num_examples):  
    """生成y=Xw+b+噪声"""
    # 均值，标准差，样本数，样本w列数
    X = torch.normal(0, 1, (num_examples, len(w))) # 生成服从正态分布的随机数。
    # X = torch.normal(0, 1, (num_examples, w.shape[0])) 
    # 线性公式生成y
    y = torch.matmul(X, w) + b
    # 噪声
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1)) #reshape((-1, 1))一千行一列的形状

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
print('features:', features[0],'\nlabel:', labels[0])

# 绘制数据散点图
d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1);
# d2l.plt.show()

# 小批量读取数据
def data_iter(batch_size, features, labels):
    # 获取一共有多少条数据（1000）
    num_examples = len(features)
    # range(1000)生成一个从0到999的列表
    indices = list(range(num_examples))
    # random.shuffle(indices)将列表随机打乱 
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):  # 开始，结束，步长
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)]) # 切分，左闭右开，防止越界
        yield features[batch_indices], labels[batch_indices] # yield返回数据，并保存当前函数的状态，下次再调用时从这里继续往下执行

batch_size = 10
# for X, y in data_iter(batch_size, features, labels):
#     print(X, '\n', y)
#     break

# 初始化模型参数
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
print('w:', w, '\nb:', b)

def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 定义优化算法
def sgd(params, lr, batch_size):    # 模型参数，学习率，数据量
    """小批量随机梯度下降"""
    with torch.no_grad():  # 不需要计算梯度
        for param in params: # params = [w, b]
            param -= lr * param.grad / batch_size  # 更新参数
            param.grad.zero_()  # 梯度清零


# 训练过程
lr = 0.03  # 学习率
num_epochs = 3  # 迭代次数
net = linreg  # 模型
loss = squared_loss  # 损失函数

for epoch in range(num_epochs): # 训练三轮
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward() # 反向传播，计算梯度，给一个批次的梯度相加求和
        sgd([w, b], lr, batch_size) # 更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        # print("train_l_shape: ", train_l.shape)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差：{true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差：{true_b - b}')