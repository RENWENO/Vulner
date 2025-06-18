#! -*- coding: utf-8 -*-
# @Time    : 2025/5/21 19:06
# @Author  : xx

import torch


def calculate_f1(preds, targets, average='macro', num_classes=None, from_logits=True):
    """
    计算 F1 值（支持多分类和二分类）

    Args:
        preds (torch.Tensor): 模型输出 logits 或预测类别，形状为 (N, C) 或 (N,)
        targets (torch.Tensor): 真实标签，形状为 (N,)
        average (str): 平均方式，可选 'micro'、'macro'、'weighted' 或 'none'
        num_classes (int): 类别数（若为 None，则自动推断）
        from_logits (bool): 输入 preds 是否为 logits（需要 argmax 转换为类别）

    Returns:
        torch.Tensor: F1 值（标量或各类别 F1 值）
    """
    # 如果输入是 logits，转换为预测类别
    if from_logits:
        preds = torch.argmax(preds, dim=1)  # (N, C) -> (N,)

    # 确保 preds 和 targets 形状一致
    assert preds.shape == targets.shape, "预测和标签形状不一致"

    # 自动推断类别数
    if num_classes is None:
        num_classes = max(torch.max(preds).item(), torch.max(targets).item()) + 1

    # 生成混淆矩阵（高效向量化实现）
    device = preds.device
    targets = targets.view(-1).long().to(device)
    preds = preds.view(-1).long().to(device)
    indices = targets * num_classes + preds
    cm_flat = torch.bincount(indices, minlength=num_classes ** 2).to(device)
    confusion_matrix = cm_flat.view(num_classes, num_classes)  # (C, C)

    # 计算 TP、FP、FN
    tp = confusion_matrix.diag()  # 对角线为 TP
    fp = confusion_matrix.sum(dim=0) - tp  # 每列和 - TP = FP
    fn = confusion_matrix.sum(dim=1) - tp  # 每行和 - TP = FN

    # 计算 Precision 和 Recall（避免除零）
    epsilon = 1e-12
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1_per_class = 2 * (precision * recall) / (precision + recall + epsilon)

    # 根据平均方式计算最终 F1
    if average == 'micro':
        total_tp = tp.sum()
        total_fp = fp.sum()
        total_fn = fn.sum()
        micro_precision = total_tp / (total_tp + total_fp + epsilon)
        micro_recall = total_tp / (total_tp + total_fn + epsilon)
        f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall + epsilon)
    elif average == 'macro':
        f1 = f1_per_class.mean()
    elif average == 'weighted':
        support = confusion_matrix.sum(dim=1).float()  # 每个类别的样本数
        f1 = (f1_per_class * support).sum() / support.sum()
    elif average == 'none':
        f1 = f1_per_class
    else:
        raise ValueError("平均方式必须是 'micro', 'macro', 'weighted' 或 'none'")

    return f1
