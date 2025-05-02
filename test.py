import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dataset import BendDataset, prepare_data
from model import TwoStageBendModel
from config import BASE_PATH, get_test_transform


def test_model(model, loader, device='cuda'):
    # model.to(device)
    # model.eval()

    # 类别名称
    position_names = ['0m', '1m', '2m', '3m', '4m']
    bend_names = ['1.25cm', '2.5cm', '3cm', '4.3cm', '无弯曲']

    correct_pos = 0
    correct_bend = 0
    total = 0

    with torch.no_grad():
        for images, pos_labels, bend_labels in loader:
            images = images.to(device)
            pos_labels = pos_labels.to(device)
            bend_labels = bend_labels.to(device)

            pos_outputs, bend_outputs = model(images)

            # 获取预测概率
            pos_probs = torch.softmax(pos_outputs, dim=1)
            bend_probs = torch.softmax(bend_outputs, dim=1)

            # 获取预测类别
            _, pos_predicted = torch.max(pos_outputs.data, 1)
            _, bend_predicted = torch.max(bend_outputs.data, 1)

            # 计算正确率
            correct_pos += (pos_predicted == pos_labels).sum().item()
            correct_bend += (bend_predicted == bend_labels).sum().item()
            total += pos_labels.size(0)

            # 打印每个样本的预测结果
            for i in range(len(pos_labels)):
                print(f"\n样本 {total - len(pos_labels) + i + 1}:")
                print(f"真实位置: {position_names[pos_labels[i]]}, 预测位置: {position_names[pos_predicted[i]]}")
                print(f"位置概率: {pos_probs[i][pos_predicted[i]].item():.4f}")
                print(f"真实弯曲: {bend_names[bend_labels[i]]}, 预测弯曲: {bend_names[bend_predicted[i]]}")
                print(f"弯曲概率: {bend_probs[i][bend_predicted[i]].item():.4f}")

    # 计算并打印准确率
    pos_accuracy = 100 * correct_pos / total
    bend_accuracy = 100 * correct_bend / total

    print("\n" + "=" * 50)
    print(f"位置分类准确率: {pos_accuracy:.2f}%")
    print(f"弯曲分类准确率: {bend_accuracy:.2f}%")
    print(f"综合准确率: {(pos_accuracy + bend_accuracy) / 2:.2f}%")
    print("=" * 50)


def main():
    # 准备数据
    image_paths, position_labels, bend_labels = prepare_data(BASE_PATH)

    _, test_paths, _, test_pos, _, test_bend = train_test_split(
        image_paths, position_labels, bend_labels,
        test_size=0.1, random_state=42
    )

    test_dataset = BendDataset(test_paths, test_pos, test_bend, get_test_transform())
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )

    # 初始化模型
    model = TwoStageBendModel()

    # 加载预训练权重
    model.load_state_dict(torch.load('two_stage_bend_model.pth'))

    # 测试模型
    print("开始测试模型...")
    test_model(model, test_loader)


if __name__ == '__main__':
    main()