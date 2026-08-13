import torch

A = torch.arange(12,dtype=torch.float32).reshape(3,4)
print(A)
B = A.clone()
print(A*B)

a = 2
C = A.clone()
print(a+C)
print(a*C)

print(C)
C_sum0 = C.sum(dim=0)
print(C_sum0)
print(C_sum0.shape)
print(C[:,0:2].sum(dim=0))
print(C[:,0:2].sum(dim=0).shape)


#计算平均值
print(A.mean())
print(A.sum()/A.numel())
print(A.mean(dim=0))
print(A.sum(dim=0)/A.shape[0])

# 广播机制
A = A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print("-------广播机制-----------")
print(A)
sum_A = A.sum(dim=1, keepdim=True) #dim=1，每一行求和
print(sum_A)
print(A / sum_A)
# A.shape      = [5, 4]
# sum_A.shape  = [5, 1]
# PyTorch 会把：
# tensor([[ 6.],
#         [22.],
#         [38.],
#         [54.],
#         [70.]])
# 在逻辑上自动“扩展”为：
# [[ 6,  6,  6,  6],
#  [22, 22, 22, 22],
#  [38, 38, 38, 38],
#  [54, 54, 54, 54],
#  [70, 70, 70, 70]]
# 最终运算就是
# [[ 0,  1,  2,  3]      [[ 6,  6,  6,  6]
#  [ 4,  5,  6,  7]       [22, 22, 22, 22]
#  [ 8,  9, 10, 11]   ÷   [38, 38, 38, 38]
#  [12, 13, 14, 15]       [54, 54, 54, 54]
#  [16, 17, 18, 19]]      [70, 70, 70, 70]]

# 累加求和，从第零维开始，也就是按行从上向下累加求和
print(A.cumsum(dim=0))

# 点积运算
print("-------点积运算-----------")
x = torch.arange(4, dtype=torch.float32)
y = torch.ones(4, dtype=torch.float32)
print(x,y, torch.dot(x, y))
print(torch.sum(x*y))
# torch.sum(input)
# torch.sum(input, dim=...)
# torch.sum(input, dim=..., keepdim=True)

# 矩阵向量积
print("-------矩阵向量积-----------")
A = torch.arange(9,dtype=torch.float32).reshape(3,3)
print(A)
x = torch.arange(3,dtype=torch.float32)
print(x)
print(A.shape, x.shape)
print(torch.mv(A,x), torch.mv(A,x).shape)
# 矩阵乘法
print("-------矩阵乘法-----------")
B = torch.ones(3,3,dtype=torch.float32)
print(B)
print(torch.mm(A,B), torch.mm(A,B).shape)