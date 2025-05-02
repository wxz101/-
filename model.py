import torch
import torch.nn as nn
from torchvision import models


class TwoStageBendModel(nn.Module):
    def __init__(self, num_positions=5, num_bend_types=5):
        super(TwoStageBendModel, self).__init__()

        # 共享的特征提取器
        self.feature_extractor = models.efficientnet_b0(pretrained=True)
        self.feature_extractor.classifier = nn.Identity()  # 移除最后的分类层

        # 第一阶段模型 - 位置识别
        self.position_classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_positions)
        )

        # 第二阶段模型 - 弯曲程度识别
        self.bend_classifier = nn.Sequential(
            nn.Linear(1280 + num_positions, 512),  # 拼接特征和位置预测
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_bend_types)
        )

    def forward(self, x, stage='both'):
        features = self.feature_extractor(x)

        if stage == 'position':
            return self.position_classifier(features)

        position_logits = self.position_classifier(features)
        position_probs = torch.softmax(position_logits, dim=1)

        if stage == 'bend':
            combined = torch.cat([features, position_probs], dim=1)
            return self.bend_classifier(combined)

        # 默认返回两个输出
        combined = torch.cat([features, position_probs], dim=1)
        bend_logits = self.bend_classifier(combined)

        return position_logits, bend_logits