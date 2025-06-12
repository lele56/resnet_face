import torch 
from torch import nn
import numpy as np
import random
import os
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

# 定义残差块
#18/34
class basicblock(nn.Module):
    expansion = 1
    # 初始化函数，定义基本块的结构
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        # 调用父类的初始化函数
        super(basicblock, self).__init__()
        # 第一个卷积层，输入通道数为in_channels，输出通道数为out_channels，卷积核大小为3x3，步长为stride，填充为1，不使用偏置
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        # 第一个批归一化层，对卷积层的输出进行归一化
        self.bn1 = nn.BatchNorm2d(out_channels)
        # ReLU激活函数，inplace=True表示直接在输入上进行操作，节省内存
        self.relu = nn.ReLU(inplace=True)
        # 第二个卷积层，输入通道数为out_channels，输出通道数为out_channels，卷积核大小为3x3，步长为1，填充为1，不使用偏置
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # 第二个批归一化层，对卷积层的输出进行归一化
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 下采样层，用于调整特征图的尺寸和通道数
        self.downsample = downsample
        # 添加Dropout层，防止过拟合
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # 保存输入的特征图作为残差连接的初始值
        identity = x
        # 如果存在下采样操作，则对输入进行下采样，用于调整特征图的尺寸和通道数，以便与卷积层的输出进行相加
        if self.downsample is not None:
            identity = self.downsample(x)
        # 第一个卷积层，对输入特征图进行卷积操作
        out = self.conv1(x)
        # 第一个批归一化层，对卷积层的输出进行归一化
        out = self.bn1(out)
        # ReLU激活函数，增加非线性特性
        out = self.relu(out)
        # 第二个卷积层，对经过第一个卷积层和激活函数处理后的特征图进行卷积操作
        out = self.conv2(out)
        # 第二个批归一化层，对第二个卷积层的输出进行归一化
        out = self.bn2(out)
        # 如果存在下采样操作，则对输入进行下采样，用于调整特征图的尺寸和通道数，以便与卷积层的输出进行相加
        if self.downsample is not None:
            identity = self.downsample(x)
        #在卷积层和激活函数处理后的特征图上添加Dropout层，防止过拟合
        out = self.dropout(out)
        # 将经过卷积和归一化处理后的特征图与残差连接相加
        out += identity
        # ReLU激活函数，增加非线性特性
        out= self.relu(out)
        # 返回处理后的特征图
        return out


class bottblock(nn.Module):
    #膨胀因子
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        # 调用父类的初始化函数
        super(bottblock, self).__init__()
        # 第一个卷积层，输入通道数为in_channels，输出通道数为out_channels，卷积核大小为1x1，步长为1，填充为1，不使用偏置
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False)
        # 第一个批归一化层，对卷积层的输出进行归一化
        self.bn1 = nn.BatchNorm2d(out_channels)
        # ReLU激活函数，inplace=True表示直接在输入上进行操作，节省内存
        self.relu = nn.ReLU(inplace=True)
        # 第二个卷积层，输入通道数为out_channels，输出通道数为out_channels，卷积核大小为3x3，步长为stride，填充为1，不使用偏置
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        # 第二个批归一化层，对卷积层的输出进行归一化
        self.bn2 = nn.BatchNorm2d(out_channels)
        # ReLU激活函数，inplace=True表示直接在输入上进行操作，节省内存
        self.relu = nn.ReLU(inplace=True)
        # 第三个卷积层，输入通道数为out_channels，输出通道数为out_channels*self.expansion，卷积核大小为1x1，步长为1，填充为1，不使用偏置
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels*self.expansion, kernel_size=1, stride=1, bias=False)
        # 第三个批归一化层，对卷积层的输出进行归一化
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        # ReLU激活函数，inplace=True表示直接在输入上进行操作，节省内存
        self.relu = nn.ReLU(inplace=True)
        # 下采样层，用于调整特征图的尺寸和通道数
        self.downsample = downsample
        # 添加Dropout层，防止过拟合
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.dropout(out)
        out += identity
        out = self.relu(out)
        return out
    

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,include_top=True):
        # 调用父类的初始化函数
        super(ResNet, self).__init__()
        # 是否包含顶层的全连接层
        self.include_top = include_top
        # 输入通道数
        self.in_channels = 64
        # 第一个卷积层，输入通道数为3，输出通道数为64，卷积核大小为7x7，步长为2，填充为3，不使用偏置
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        # 第一个批归一化层，对卷积层的输出进行归一化
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        # ReLU激活函数，inplace=True表示直接在输入上进行操作，节省内存
        self.relu = nn.ReLU(inplace=True)
        # 最大池化层，用于减小特征图的尺寸
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 第一个残差块层，使用basicblock，输出通道数为64，步长为1
        self.layer1 = self._make_layer(basicblock, 64, num_blocks[0], stride=1)
        # 第二个残差块层，使用basicblock，输出通道数为128，步长为2
        self.layer2 = self._make_layer(basicblock, 128, num_blocks[1], stride=2)
        # 第三个残差块层，使用bottblock，输出通道数为256，步长为2
        self.layer3 = self._make_layer(bottblock, 256, num_blocks[2], stride=2)
        # 第四个残差块层，使用bottblock，输出通道数为512，步长为2
        self.layer4 = self._make_layer(bottblock, 512, num_blocks[3], stride=2)
        # 如果包含顶层的全连接层
        if self.include_top:
            # 平均池化层，用于将特征图的尺寸减小到1x1
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            # 全连接层，用于分类，输入特征数为512*block.expansion，输出类别数为num_classes
            self.fc = nn.Linear(512*block.expansion, num_classes)

        # 对模型中的卷积层进行参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        # 初始化下采样层为None
        downsample = None
        # 如果步长不为1或者输入通道数与输出通道数*膨胀因子不相等，则需要下采样
        if stride != 1 or self.in_channels != out_channels*block.expansion:
            # 定义下采样层，包含一个1x1卷积层和一个批归一化层
            downsample = nn.Sequential(
                # 卷积层，用于调整特征图的通道数
                nn.Conv2d(self.in_channels, out_channels*block.expansion, kernel_size=1, stride=stride, bias=False),
                # 批归一化层，用于归一化特征图
                nn.BatchNorm2d(out_channels*block.expansion)
            )
        # 初始化层列表
        layers = []
        # 添加第一个残差块，传入输入通道数、输出通道数、下采样层和步长
        layers.append(block(self.in_channels, out_channels, downsample=downsample, stride=stride))
        # 更新输入通道数为输出通道数*膨胀因子
        self.in_channels = out_channels*block.expansion
        # 添加剩余的残差块，传入输入通道数和输出通道数
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        # 返回包含所有残差块的序列
        return nn.Sequential(*layers)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x
    

def resnet18(num_class,include_top=True):
    return ResNet(basicblock, [2,2,2,2], num_class,include_top=include_top)

def resnet34(num_class,include_top=True):
    return ResNet(basicblock, [3,4,6,3], num_class,include_top=include_top)

def resnet50(num_class,include_top=True):
    return ResNet(bottblock, [3,4,6,3], num_class,include_top=include_top)

def resnet101(num_class,include_top=True):
    return ResNet(bottblock, [3,4,23,3], num_class,include_top=include_top)

def resnet152(num_class,include_top=True):
    return ResNet(bottblock, [3,8,36,3], num_class,include_top=include_top)

#定义种子函数
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 遍历图像路径和标签的函数
def load_image_paths_and_labels(image_dir):
    image_paths = []  # 初始化
    labels = []
    label_map = {}
    label_idx = 0

    for root, dirs, files in os.walk(image_dir):  # 通过 os.walk 遍历目录结构
        for file in files:
            if file[-4:] in ['.jpg', '.png', '.PNG', '.JPG']:
                full_image_path = os.path.join(root, file)
                label_folder_name = os.path.basename(root)
                if label_folder_name not in list(label_map.values()):
                    label_map[label_idx] = label_folder_name
                    label_idx += 1
                try:  # 查看是否正确映射
                    label = list(label_map.keys())[list(label_map.values()).index(label_folder_name)]
                    image_paths.append(full_image_path)
                    labels.append(label)
                    # print(f"文件路径: {full_image_path}, 对应的标签: {label}")
                except ValueError:
                    print(
                        f"在获取 {full_image_path} 的标签时出现问题，文件夹名称 {label_folder_name} 在映射中不存在，请检查目录结构或映射关系。")
                except Exception as e:
                    print(f"处理 {full_image_path} 的标签添加时出现未知错误: {str(e)}")
    return image_paths, labels, label_map



class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, augmented=None):  # 构造函数
        self.image_paths = image_paths
        self.labels = labels
        self.augmented = augmented
        self.transform = transform

    def __len__(self):  # 长度
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")
        
        if self.augmented:
            img = self.augmented(image=np.array(img))['image']
            img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, label

# 定义训练函数
def train(net, train_loader,test_loader,val_loader,optimizer,scheduler, criterion, num_epochs, device):
    best_accuracy = 0.0
    train_losses = []
    val_losses = []
    test_losses = []
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
    for epoch in range(num_epochs):
        net.train()  # 将网络模型设置为训练模式
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader,desc=f'epoch:{epoch+1}/{num_epochs}',unit='batch'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # 梯度清零
            outputs = net(images)  # 前向传播
            loss = criterion(outputs, labels) # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            running_loss += loss.item()*images.size(0)
            total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
        epoch_loss = running_loss / len(train_loader.dataset)
        accuracy = correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}',f'Accuracy: {accuracy:.4f}')
        test_loss,test_accuracy =evaluate_accuracy(net, test_loader, criterion,device,data_loader_name='test') #计算测试集的准确度和误差
        val_loss,val_accuracy = evaluate_accuracy(net, val_loader, criterion,device,data_loader_name='val')  # 计算验证集准确度和误差
        # 更新验证集准确率最高的epoch
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            save_model(net, 'best_model.pth')
        scheduler.step()
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        train_accuracies.append(accuracy)
        val_accuracies.append(val_accuracy)
        test_accuracies.append(test_accuracy)
    # 输出整个训练过程中在验证集上取得的最高准确率以及对应的轮次信息
    print(f'Best Val Acc: {best_accuracy:.4f} at epoch {best_epoch + 1}')

        # 绘制图表
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()





def evaluate_accuracy(net, data_loader, criterion, device, data_loader_name=''):
    net.eval()  # 将网络模型设置为评估模式
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():  # 不需要计算梯度
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = running_loss / len(data_loader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy


def save_model(net, path):
    torch.save(net.state_dict(), path)


