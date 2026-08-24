import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
# 训练集，测试集(Dataloader对象)
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 图像[28,28]展平成784,输出十个类别的置信度
num_inputs, num_outputs = 784, 10
# 创建一个 784 × 10 的矩阵w, 取值按正态分布随机，输入x为[1,784](展平后的图像),wx为[1,10]即10个类别的置信度
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
print(f'W.shape:{W.shape}, b.shape:{b.shape}')

X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# print(X.sum(0, keepdim=True), X.sum(1, keepdim=True))

def softmax(X):
    X_exp = torch.exp(X) # X_exp为[2,3]，每一行代表一个样本
    partition = X_exp.sum(1, keepdim=True) # partition为[2,1],因为按列相加了
    return X_exp / partition # 广播机制

# 实现softmax回归模型
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

def net_1(X):
    """
    实现 Softmax 回归模型
    参数：
        X：输入的一批 Fashion-MNIST 图片
           原始形状通常为：
           [batch_size, 1, 28, 28]
    返回：
        每张图片属于 10 个类别的概率
        输出形状为：
        [batch_size, 10]
    """
    # --------------------------------------------------
    # 第 1 步：把每张 28×28 的图片展开成长度为 784 的一维向量
    # --------------------------------------------------
    # W.shape = [784, 10]
    # 所以 W.shape[0] = 784
    #
    # X 原来的形状例如：
    # [256, 1, 28, 28]
    #
    # X.reshape((-1, W.shape[0]))
    # 等价于：
    # X.reshape((-1, 784))
    #
    # reshape 后：
    # 256*1*28*28/784 = 256
    # [256, 1, 28, 28] -> [256, 784]
    #
    # -1 表示：
    # “这一维不用我指定，让 PyTorch 根据总元素个数自动计算”
    X = X.reshape((-1, W.shape[0]))
    # --------------------------------------------------
    # 第 2 步：进行线性变换 XW + b
    # --------------------------------------------------
    # X.shape = [256, 784]
    # W.shape = [784, 10]
    #
    # torch.matmul(X, W) 是矩阵乘法：
    #
    # [256, 784] × [784, 10]
    #              ↓
    #          [256, 10]
    #
    # 意思是：
    # 每张图片原来有 784 个像素特征，
    # 经过 W 的计算之后，得到 10 个类别的分数。
    #
    # 例如某张图片可能得到：
    # [1.2, -0.3, 2.5, 0.8, ...]
    #
    # 这 10 个数字现在还只是“原始分数”，还不是概率。
    Y = torch.matmul(X, W)
    # --------------------------------------------------
    # 第 3 步：加上偏置 b
    # --------------------------------------------------

    # b.shape = [10]
    #
    # Y.shape = [256, 10]
    #
    # 执行：
    # [256, 10] + [10]
    #
    # PyTorch 会使用广播机制，
    # 把同一个 b 加到每一个样本的 10 个类别分数上。
    #
    # 所以这里完成的就是经典线性模型公式：
    #
    #          Y = XW + b
    #
    Y = Y + b


    # --------------------------------------------------
    # 第 4 步：使用 Softmax 把 10 个原始分数转换成概率
    # --------------------------------------------------

    # softmax 会对每一个样本单独计算。
    #
    # 例如：
    #
    # 原始分数：
    # [1.0, 2.0, 3.0]
    #
    # softmax 后：
    # [0.09, 0.245, 0.665]
    #
    # 特点：
    # 1. 每个数字都在 0~1 之间
    # 2. 一行所有概率之和 = 1
    #
    # 对 Fashion-MNIST 来说：
    # 每一行的 10 个数字，
    # 就表示这张图片属于 10 个类别的概率。
    Y = softmax(Y)
    # 返回最终预测概率
    return Y

# y的形状为[0,2]
y = torch.tensor([0, 2])
# y_hat包含两个样本的3个类别的预测概率
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# 成对取索引，y_hat[[0,1], [0,2]]取出来[0,0],[1,2]
print(y_hat[[0, 1], y])

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])

print(cross_entropy(y_hat, y)) 


def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1) # 找到最大值所在的位置，返回索引
        # 第0维 → 样本
        # 第1维 → 类别
        # axis=1表示在类别这一维度找最大值
        # 此时y_hat为[2,2]
    cmp = y_hat.type(y.dtype) == y # 例如此时cmp为[false, true](按照上面定义的y_hat和y)
    return float(cmp.type(y.dtype).sum()) #此时cmp为[0,1],相加为1返回，即预测正确的个数

print(f'准确率：{accuracy(y_hat, y)/len(y)}')


def evaluate_accuracy(net, data_iter):
    """计算模型在数据集上的准确率"""
    if isinstance(net, torch.nn.Module):
        net.eval()
    # 创建两个“累加器”，metric[0] = 0，metric[1] = 0
    # 记录 metric[0]：预测正确的样本数量
    # metric[1]：总样本数量
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        # metric[0] += accuracy(net(X), y) 预测正确的数量
        # metric[1] += y.numel() 一共有多少个元素
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0]/metric[1]


print(f'训练集准确率：{evaluate_accuracy(net, train_iter)}')
print(f'测试集准确率：{evaluate_accuracy(net, test_iter)}')

    