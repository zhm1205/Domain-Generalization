"""
Utility functions and helper classes
"""

import torch
import torch.nn.functional as F
from typing import Tuple, List, Union, Literal


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update statistics
        
        Args:
            val: New value
            n: Number of samples for the new value
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> List[torch.Tensor]:
    """
    Compute accuracy
    
    Args:
        output: Model output [batch_size, num_classes]
        target: True labels [batch_size]
        topk: Compute top-k accuracy
        
    Returns:
        List[torch.Tensor]: List of top-k accuracies
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        # 获取top-k预测
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        
        return res


def compute_confusion_matrix(output: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    计算混淆矩阵
    
    Args:
        output: 模型输出 [batch_size, num_classes]
        target: 真实标签 [batch_size]
        num_classes: 类别数
        
    Returns:
        torch.Tensor: 混淆矩阵 [num_classes, num_classes]
    """
    with torch.no_grad():
        _, pred = torch.max(output, 1)
        
        # 创建混淆矩阵
        confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
        
        for t, p in zip(target.view(-1), pred.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
        
        return confusion_matrix


def compute_class_accuracy(confusion_matrix: torch.Tensor) -> torch.Tensor:
    """
    从混淆矩阵计算每个类别的准确率
    
    Args:
        confusion_matrix: 混淆矩阵 [num_classes, num_classes]
        
    Returns:
        torch.Tensor: 每个类别的准确率 [num_classes]
    """
    class_correct = confusion_matrix.diag()
    class_total = confusion_matrix.sum(1)
    
    # 避免除零
    class_total = torch.max(class_total, torch.ones_like(class_total))
    
    return class_correct.float() / class_total.float()



def set_random_seed(seed: int, mode: Literal["off", "practical", "strict"] = "practical") -> None:
    """
    Reproducibility presets:
      - "off":       速度优先，允许非确定性
      - "practical": 实用可复现（推荐），不调用 use_deterministic_algorithms
      - "strict":    严格确定性，需要 CUBLAS_WORKSPACE_CONFIG，可能更慢
    """
    import os, random, numpy as np, torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if mode == "off":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return

    if mode == "practical":
        # practical reproducibility, common and stable
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return

    if mode == "strict":
        # strict reproducibility: requires setting this environment variable early in the process (the earlier the better)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)
        return

    raise ValueError("mode must be 'off' | 'practical' | 'strict'")


def save_tensor(tensor: torch.Tensor, path: str) -> None:
    """
    保存张量到文件
    
    Args:
        tensor: 要保存的张量
        path: 保存路径
    """
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(tensor, path)


def load_tensor(path: str, device: torch.device = None) -> torch.Tensor:
    """
    从文件加载张量
    
    Args:
        path: 文件路径
        device: 目标设备
        
    Returns:
        torch.Tensor: 加载的张量
    """
    if device is None:
        device = torch.device("cpu")
    
    return torch.load(path, map_location=device)


def mean_average_precision(query_features: torch.Tensor, 
                          gallery_features: torch.Tensor,
                          query_labels: torch.Tensor,
                          gallery_labels: torch.Tensor) -> float:
    """
    Compute mean Average Precision (mAP) for ReID
    
    Args:
        query_features: Query features [num_queries, feature_dim]
        gallery_features: Gallery features [num_gallery, feature_dim]  
        query_labels: Query labels [num_queries]
        gallery_labels: Gallery labels [num_gallery]
        
    Returns:
        float: mAP score
    """
    # Compute distance matrix
    distances = torch.cdist(query_features, gallery_features)
    
    # Sort by distance (ascending)
    indices = torch.argsort(distances, dim=1)
    
    # Compute AP for each query
    aps = []
    for i in range(query_features.size(0)):
        query_label = query_labels[i]
        ranked_labels = gallery_labels[indices[i]]
        
        # Find matches
        matches = (ranked_labels == query_label).float()
        
        # Compute AP
        if matches.sum() == 0:
            continue
            
        precisions = []
        num_matches = 0
        for j, match in enumerate(matches):
            if match == 1:
                num_matches += 1
                precision = num_matches / (j + 1)
                precisions.append(precision)
        
        if precisions:
            ap = sum(precisions) / len(precisions)
            aps.append(ap)
    
    return sum(aps) / len(aps) if aps else 0.0


def compute_rank_accuracy(query_features: torch.Tensor,
                         gallery_features: torch.Tensor, 
                         query_labels: torch.Tensor,
                         gallery_labels: torch.Tensor,
                         k: int = 1) -> float:
    """
    Compute Rank-k accuracy for ReID
    
    Args:
        query_features: Query features [num_queries, feature_dim]
        gallery_features: Gallery features [num_gallery, feature_dim]
        query_labels: Query labels [num_queries] 
        gallery_labels: Gallery labels [num_gallery]
        k: Rank threshold
        
    Returns:
        float: Rank-k accuracy
    """
    # Compute distance matrix
    distances = torch.cdist(query_features, gallery_features)
    
    # Sort by distance (ascending) and get top-k
    indices = torch.argsort(distances, dim=1)[:, :k]
    
    # Check if query label appears in top-k
    correct = 0
    for i in range(query_features.size(0)):
        query_label = query_labels[i]
        top_k_labels = gallery_labels[indices[i]]
        if query_label in top_k_labels:
            correct += 1
    
    return correct / query_features.size(0)
