# 读取csv文件数据转换为tensor张量
import os
import pandas as pd
import torch


#创建文件夹
os.makedirs(os.path.join("..", "data"), exist_ok=True)
#创建文件夹下的文件
data_file = os.path.join("..", "data", "house_tiny.csv")
with open(data_file,"w") as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')


data = pd.read_csv(data_file)
print(data)
# 拆分输入输出
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
print("inputs:\n", inputs)
inputs = inputs.fillna(inputs.mean(numeric_only=True))
print("inputs_after_fillina:\n", inputs)
inputs = pd.get_dummies(inputs, dummy_na=True)
print("inputs_after_get_dummies:\n", inputs)

print("outputs:\n", outputs)

X, y = torch.tensor(inputs.to_numpy(dtype=float)), torch.tensor(outputs.to_numpy(dtype=float))
print("X:\n", X)
print("y:\n", y)