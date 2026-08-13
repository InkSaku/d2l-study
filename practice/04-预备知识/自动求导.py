import torch

x = torch.arange(4, dtype=torch.float32)
print(x)
# 告诉pytorch这个张量需要计算梯度,记录计算过程。（requires_grad_()是原地操作，直接改变x）
x.requires_grad_(True)
print(x.grad)

y = 2 * torch.dot(x, x) # x内积乘2
print(y) # grad_fn=<MulBackward0>，建立了一个计算图，用于反向传播求导
y.backward() # 反向传播，计算梯度
print(x.grad, x.grad==4*x) # 输出梯度值

x.grad.zero_() # 梯度清零
y = x.sum()
print(y)
y.backward()
print(x.grad)

x.grad.zero_()
y = x*x
u = y.detach() # 生成一个和y数值相同的张量u，但是u不再是一个关于x的张量，
z = u*x
z.sum().backward()
print(x.grad==u)