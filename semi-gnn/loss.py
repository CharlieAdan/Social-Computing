import torch
import torch.nn.functional as F

def focal_loss(output, labels, alpha=0.25, gamma=2.0, weight=None):
    """
    Focal Loss implementation for imbalanced classification.
    output: log_softmax output from model
    labels: true labels
    """
    # output is log_softmax, so exp(output) is softmax probability
    probs = torch.exp(output)
    
    # Gather the probabilities of the true classes
    # labels.view(-1, 1) makes it a column vector to gather from probs
    pt = probs.gather(1, labels.view(-1, 1)).view(-1)
    
    log_pt = output.gather(1, labels.view(-1, 1)).view(-1)
    
    # Calculate focal weight
    # If weight (class_weights) is provided, we can combine it or just use alpha
    # Here we use alpha for the positive class (1) and 1-alpha for negative (0)
    # Assuming label 1 is minority/positive
    
    # Standard Focal Loss formulation: -alpha_t * (1-pt)^gamma * log(pt)
    # We'll use the alpha parameter to balance classes if weight is not provided
    
    if weight is not None:
        # If explicit class weights are provided, use them as alpha_t
        alpha_t = weight.gather(0, labels)
    else:
        # Simple alpha balancing
        alpha_t = torch.where(labels == 1, alpha, 1-alpha)
        
    loss = -alpha_t * (1 - pt) ** gamma * log_pt
    return loss.mean()

def loss_function(output, labels, embedding, edge_index, train_mask, reg_lambda=0.01, class_weights=None, use_focal=False):
    """
    双重损失函数：分类损失 + 关系损失

    Args:
        output (torch.Tensor): 模型的分类输出 (log_softmax)。
        labels (torch.Tensor): 真实标签。
        embedding (torch.Tensor): 节点的嵌入表示。
        edge_index (torch.Tensor): 边索引。
        train_mask (torch.Tensor): 训练集掩码。
        reg_lambda (float): 关系损失的权重系数。
        class_weights (torch.Tensor, optional): 类别权重，用于处理类别不平衡。
        use_focal (bool): 是否使用 Focal Loss。

    Returns:
        torch.Tensor: 总损失。
        torch.Tensor: 分类损失。
        torch.Tensor: 关系损失。
    """
    # 1. 分类损失 (Classification Loss)
    # 只计算有标签节点的损失
    masked_output = output[train_mask]
    masked_labels = labels[train_mask]
    
    if use_focal:
        # 使用 Focal Loss 处理极度不平衡
        # alpha=0.75 意味着我们更关注正样本(1)，gamma=2.0 关注难分样本
        classification_loss = focal_loss(masked_output, masked_labels, alpha=0.75, gamma=2.0, weight=class_weights)
    elif class_weights is not None:
        classification_loss = F.nll_loss(masked_output, masked_labels, weight=class_weights)
    else:
        classification_loss = F.nll_loss(masked_output, masked_labels)

    # 2. 关系损失 (Relationship/Consistency Loss)
    # 目标：让相连的节点在嵌入空间中更接近
    # 适用于所有节点（有标签和无标签）
    start_nodes, end_nodes = edge_index
    
    # 获取相连节点的嵌入
    emb_start = embedding[start_nodes]
    emb_end = embedding[end_nodes]
    
    # 计算它们之间的L2距离的平方
    distance = torch.sum((emb_start - emb_end) ** 2, dim=1)
    relationship_loss = torch.mean(distance)

    # 总损失
    total_loss = classification_loss + reg_lambda * relationship_loss
    
    return total_loss, classification_loss, relationship_loss
