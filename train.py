import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import BendDataset, prepare_data, split_data
from model import TwoStageBendModel
from config import BASE_PATH, TRAIN_PARAMS, get_train_transform, get_val_transform


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5, device='cuda'):
    model.to(device)
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, pos_labels, bend_labels in train_loader:
            images = images.to(device)
            pos_labels = pos_labels.to(device)
            bend_labels = bend_labels.to(device)

            optimizer.zero_grad()

            # 训练位置分类器
            pos_outputs = model(images, stage='position')
            pos_loss = criterion(pos_outputs, pos_labels)

            # 训练弯曲分类器
            bend_outputs = model(images, stage='bend')
            bend_loss = criterion(bend_outputs, bend_labels)

            # 组合损失
            total_loss = pos_loss + bend_loss
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

            # 计算准确率
            _, pos_predicted = torch.max(pos_outputs.data, 1)
            _, bend_predicted = torch.max(bend_outputs.data, 1)
            correct += (pos_predicted == pos_labels).sum().item()
            correct += (bend_predicted == bend_labels).sum().item()
            total += pos_labels.size(0) * 2  # 两个任务

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # 验证
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
        print('-' * 20)

    return history


def evaluate_model(model, loader, criterion, device='cuda'):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, pos_labels, bend_labels in loader:
            images = images.to(device)
            pos_labels = pos_labels.to(device)
            bend_labels = bend_labels.to(device)

            pos_outputs, bend_outputs = model(images)

            pos_loss = criterion(pos_outputs, pos_labels)
            bend_loss = criterion(bend_outputs, bend_labels)
            total_loss = pos_loss + bend_loss

            running_loss += total_loss.item()

            _, pos_predicted = torch.max(pos_outputs.data, 1)
            _, bend_predicted = torch.max(bend_outputs.data, 1)
            correct += (pos_predicted == pos_labels).sum().item()
            correct += (bend_predicted == bend_labels).sum().item()
            total += pos_labels.size(0) * 2

    loss = running_loss / len(loader)
    acc = 100 * correct / total

    return loss, acc


def plot_training_curves(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()


def main():
    # 准备数据
    image_paths, position_labels, bend_labels = prepare_data(BASE_PATH)

    # 划分数据集
    (train_paths, train_pos, train_bend), \
        (val_paths, val_pos, val_bend), \
        (test_paths, test_pos, test_bend) = split_data(
        image_paths, position_labels, bend_labels,
        test_size=TRAIN_PARAMS['test_size'],
        val_size=TRAIN_PARAMS['val_size']
    )

    # 创建数据集和数据加载器
    train_dataset = BendDataset(train_paths, train_pos, train_bend, get_train_transform())
    val_dataset = BendDataset(val_paths, val_pos, val_bend, get_val_transform())

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_PARAMS['batch_size'],
        shuffle=True,
        num_workers=TRAIN_PARAMS['num_workers']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_PARAMS['batch_size'],
        shuffle=False,
        num_workers=TRAIN_PARAMS['num_workers']
    )

    # 初始化模型
    model = TwoStageBendModel()

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=TRAIN_PARAMS['learning_rate'])

    # 训练模型
    history = train_model(
        model, train_loader, val_loader,
        criterion, optimizer,
        num_epochs=TRAIN_PARAMS['num_epochs']
    )

    # 保存模型
    torch.save(model.state_dict(), 'two_stage_bend_model.pth')

    # 绘制训练曲线
    plot_training_curves(history)


if __name__ == '__main__':
    main()