import os
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from config import BEND_TYPES, POSITIONS


class BendDataset(Dataset):
    def __init__(self, image_paths, position_labels, bend_labels, transform=None):
        self.image_paths = image_paths
        self.position_labels = position_labels
        self.bend_labels = bend_labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        position_label = self.position_labels[idx]
        bend_label = self.bend_labels[idx]

        return image, position_label, bend_label


def prepare_data(base_path):
    image_paths = []
    position_labels = []
    bend_labels = []

    # 处理有弯曲的情况
    for bend_idx, bend_type in enumerate(BEND_TYPES[:-1]):
        for pos_idx, position in enumerate(POSITIONS):
            folder_path = os.path.join(base_path, bend_type, position, 'MER-231-41U3M-L(NM0200070009)')
            if os.path.exists(folder_path):
                for img_file in os.listdir(folder_path):
                    if img_file.lower().endswith(('.jpg', '.png')):
                        image_paths.append(os.path.join(folder_path, img_file))
                        position_labels.append(pos_idx)
                        bend_labels.append(bend_idx)

    # 处理无弯曲的情况
    no_bend_path = os.path.join(base_path, '无弯曲', 'MER-231-41U3M-L(NM0200070009)')
    if os.path.exists(no_bend_path):
        for img_file in os.listdir(no_bend_path):
            if img_file.lower().endswith(('.jpg', '.png')):
                image_paths.append(os.path.join(no_bend_path, img_file))
                position_labels.append(2)  # 无弯曲默认位置为2m
                bend_labels.append(4)  # 无弯曲标签

    return image_paths, position_labels, bend_labels


def split_data(image_paths, position_labels, bend_labels, test_size=0.2, val_size=0.5, random_state=42):
    # 首先划分训练集和临时集（验证+测试）
    train_paths, temp_paths, train_pos, temp_pos, train_bend, temp_bend = train_test_split(
        image_paths, position_labels, bend_labels,
        test_size=test_size, random_state=random_state
    )

    # 然后从临时集中划分验证集和测试集
    val_paths, test_paths, val_pos, test_pos, val_bend, test_bend = train_test_split(
        temp_paths, temp_pos, temp_bend,
        test_size=val_size, random_state=random_state
    )

    return (train_paths, train_pos, train_bend), (val_paths, val_pos, val_bend), (test_paths, test_pos, test_bend)