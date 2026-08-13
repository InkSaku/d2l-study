import torch 

x = torch.arange(12)

shape = x.shape
print(shape)

print(x.numel())

# 不改变内容，只改变张量形状
x = x.reshape(3,4)
print(x)

print(torch.zeros(2,3,4))
print(torch.ones(3,4))

y = torch.tensor([[[1,2,3,4],[5,6,7,8],[9,10,11,12]],
                  [[1,2,3,4],[5,6,7,8],[9,10,11,12]],
                  [[1,2,3,4],[5,6,7,8],[9,10,11,12]],
                  ])
print(y.shape)

print(x==torch.ones(3,4))
print(x.sum())
print(x[-1,])
print(x[1:3,]) # [1,3)取第一行第二行，左开右闭区间

before_id = id(x)
x += x # 原地运算
x[:] = x+x # 原地运算
print(id(x) == before_id)
x = x+x # 分配新内存
print(id(x) == before_id)

z = torch.zeros_like(x)
before_z = id(z)
print('id(z):', before_z)
z[:] = x+x
print('id(z):', id(z))
print(id(z)==before_z)

#广播机制