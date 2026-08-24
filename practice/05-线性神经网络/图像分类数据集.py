import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()

trans = transforms.ToTensor() # 读取图片转换成tensor格式，才可以进入神经网络
# 取出训练集
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
# 取出测试集
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)

# print(len(mnist_train), len(mnist_test))
# minist_train由60000个[图片，标签]组成，
print(mnist_train[0][0].shape) # mnist_train[0][0]是第一个图片，mnist_train[0][1]是第一个图片的标签

def get_fashion_mnist_labels(labels):  
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  
    # 图片列表，图片显示多少行，图片显示多少列，标题，图片大小
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale) # 计算画布大小
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize) # 第一个返回值（总画布）不要，只要第二个返回值axes
    axes = axes.flatten() # 展平axes
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        # 去掉坐标轴
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        # 设置标题
        if titles:
            ax.set_title(titles[i])
    return axes

X, y = next(iter(data.DataLoader(mnist_train, batch_size=9)))
show_images(X.reshape(9, 28, 28), 3, 3, titles=get_fashion_mnist_labels(y));
d2l.plt.show()

batch_size = 256

def get_dataloader_workers():  #@save
    """使用4个进程来读取数据"""
    return 4

# 读取训练集对象
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers()) # 用多少个子进程帮助读取和准备数据
# 看看数据读取要多少秒
timer = d2l.Timer()
for X, y in train_iter:
    continue
print(f'{timer.stop():.2f} sec')

def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize)) # 在trans列表起始位置插入Resize对象
    # 列表里的多个图片处理操作，按照顺序组合成一个完整的处理流程
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))