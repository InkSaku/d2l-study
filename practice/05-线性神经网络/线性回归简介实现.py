import numpy as np
import torch
from d2l import torch as d2l
from torch.utils import data


true_w = torch.tensor([2,-3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""

    # 将传入的数据（如 features 和 labels）按相同下标组合成一个数据集
    # *data_arrays 表示将元组解包
    # 例如 data_arrays = (features, labels)
    # 等价于 data.TensorDataset(features, labels)
    dataset = data.TensorDataset(*data_arrays)

    # 创建数据加载器
    # dataset：要读取的数据集
    # batch_size：每次读取多少条数据
    # shuffle=is_train：
    #   is_train=True  -> shuffle=True，训练时随机打乱数据
    #   is_train=False -> shuffle=False，测试/验证时不打乱数据
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array([features, labels], batch_size)
# iter(data_iter)：把 DataLoader 转成迭代器
# next(...)：从迭代器中取出下一批数据
next(iter(data_iter))

from torch import nn
net = nn.Sequential(nn.Linear(2, 1)) # 输入维度，输出维度
# net[0]表示网络的第一层
net[0].weight.data.normal_(0, 0.01) # 权重初始化为正态分布的一个值
net[0].bias.data.fill_(0) # 偏置初始化为0

# 计算均方误差
loss = nn.MSELoss()
# 随机梯度下降
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
# 开始训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad() # 清空上一轮（batch)计算留下来的梯度
        l.backward() # 计算本轮梯度
        trainer.step() # 更新参数
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
