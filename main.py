import argparse
import torch
from torch import nn
import os
from tqdm import tqdm
import time
import sys
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from imgaug import augmenters as iaa
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from model import set_seed,train,resnet50,load_image_paths_and_labels,CustomImageDataset,resnet152,resnet101
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#创建参数
parser = argparse.ArgumentParser()
# 定义参数
parser.add_argument('--seed', type=int, default=66, help='random seed')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--num_epochs', type=int, default= 50 , help='number of epochs')
parser.add_argument('--data_dir', type=str, default='data', help='data directory')
parser.add_argument('--model', type=str, default='resnet50', help='model name')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--num_classes', type=int, default=12, help='number of classers')
# 解析参数
args = parser.parse_args()
#确保结果可复现
set_seed(args.seed)

#对图像进行预处理
tarin_augmented =A.Compose([
        A.RandomRotate90(),  # 随机旋转图像，角度范围为90度
        A.RandomResizedCrop(height=224, width=224),  # 随机裁剪图像为224x224大小
        A.HorizontalFlip(p=0.5),  # 随机水平翻转图像
        A.VerticalFlip(p=0.5),  # 随机垂直翻转图像
        #A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),  # 颜色抖动
        #A.GaussianBlur(blur_limit=(3, 7), p=0.5),  # 高斯模糊
        #A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),  # 高斯噪声
        #A.CoarseDropout(max_holes=8, max_height=8, max_width=8, min_holes=4, min_height=4, min_width=4, p=0.5),  # 随机擦除
        A.FancyPCA(alpha=0.1, p=0.5)])  # Fancy PCA
train_transformer =  transforms.Compose([
        transforms.Resize([256,256]),  # 调整图像大小为256x256
        transforms.ToTensor(),  # 将图像转换为张量，
        transforms.Normalize((0.3738,0.3738,0.3738),(0.3240,0.3240,0.3240))])  # 对图像进行标准化
test_transformer =  transforms.Compose([
        transforms.Resize([224,224]),  # 调整图像大小为256x256
        transforms.ToTensor(),      # 把图片转化为tensor张量
        transforms.Normalize((0.3738,0.3738,0.3738),(0.3240,0.3240,0.3240))])  # 对图像进行标准化

#定义数据集加载类
# 加载图像数据并划分数据集
image_dir = args.data_dir
image_paths, labels, label_map = load_image_paths_and_labels(image_dir)

# 8:1:1划分数据集：第一次划分，将整个数据集划分为训练集和测试集；第二次划分，在训练集内部进一步划分出验证集
train_paths, test_paths, train_labels, test_labels = train_test_split(image_paths, labels, test_size=0.2,
                                                                      random_state=42)
train_paths, val_paths, train_labels, val_labels = train_test_split(train_paths, train_labels, test_size=0.125,
                                                                    random_state=42)  # 0.125 * 0.8 = 0.1

# 创建数据集
train_dataset = CustomImageDataset(train_paths, train_labels, transform=train_transformer,augmented=tarin_augmented)
val_dataset = CustomImageDataset(val_paths, val_labels, transform=test_transformer)
test_dataset = CustomImageDataset(test_paths, test_labels, transform=test_transformer)

# 创建数据加载器
batch_size = args.batch_size
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True,pin_memory=True)  # shuffle为True，打乱数据集
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, shuffle=True,pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=False,pin_memory=True)

# 根据参数选择模型
if args.model == 'resnet50':
    net = resnet50(args.num_classes).to(device)
elif args.model == 'resnet101':
    net = resnet101(args.num_classes).to(device)
elif args.model == 'resnet152':
    net = resnet152(args.num_classes).to(device)
else:
    raise ValueError(f"Invalid model name: {args.model}")

# 创建优化器和损失函数对象
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

# 创建学习率调度器
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0)

# 调用训练函数开始训练

train(net, train_loader,test_loader,val_loader, optimizer,scheduler, criterion, args.num_epochs, device)
