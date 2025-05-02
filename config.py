import os
from torchvision import transforms

# 数据路径配置
BASE_PATH = r"C:\Users\wxz_1\Desktop\测试的测试（256规格）\测试的测试（256规格）\测试"

# 数据类别配置
BEND_TYPES = ['1.25cm', '2.5cm', '3cm', '4.3cm', '无弯曲']
POSITIONS = ['0m', '1m', '2m', '3m', '4m']

# 训练参数配置
TRAIN_PARAMS = {
    'batch_size': 32,
    'num_workers': 4,
    'learning_rate': 1e-4,
    'num_epochs': 5,
    'test_size': 0.2,
    'val_size': 0.5  # 从测试集中划分验证集的比例
}

# 数据增强配置
def get_train_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_val_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_test_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])