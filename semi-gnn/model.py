# -*- coding: utf-8 -*-
"""
@File: semi_gnn_model.py
@Time: 2025/11/23
@Description: 多视图半监督图神经网络模型
本模块定义了 Semi-GNN 模型的完整结构，包括：
1.  基于 GCN 的基础图卷积网络。
2.  多视图特征融合模块，包含节点级和视图级两层注意力机制。
3.  模型的前向传播逻辑。
4.  双重损失函数（分类损失 + 关系一致性损失）的定义。

核心创新点：
-   **多视图注意力 (Multi-view Attention)**: 动态学习交易、传播、社交三类特征的重要性。
-   **半监督学习 (Semi-supervised Learning)**: 同时利用有标签和无标签数据进行训练，通过关系损失约束无标签节点的表示。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse

class ViewAttention(nn.Module):
    """
    视图级注意力机制
    对不同视图（交易、传播、社交）的特征进行加权融合。
    """
    def __init__(self, input_dim, num_views):
        super(ViewAttention, self).__init__()
        self.num_views = num_views
        self.attention_mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, view_embeddings):
        """
        Args:
            view_embeddings (list of torch.Tensor): 每个元素是 shape [num_nodes, hidden_dim] 的视图嵌入。

        Returns:
            torch.Tensor: 融合后的节点嵌入, shape [num_nodes, hidden_dim]。
            torch.Tensor: 视图注意力权重, shape [num_views]。
        """
        # 将视图嵌入堆叠起来
        stacked_views = torch.stack(view_embeddings, dim=1) # [num_nodes, num_views, hidden_dim]
        
        # 计算每个视图的注意力得分
        # 为了得到每个视图的全局重要性，我们先对节点维度取平均
        avg_view_repr = torch.mean(stacked_views, dim=0) # [num_views, hidden_dim]
        
        attention_scores = self.attention_mlp(avg_view_repr) # [num_views, 1]
        attention_weights = self.softmax(attention_scores.t()) # [1, num_views]

        # 将注意力权重扩展为 [1, num_views, 1] 以便与 stacked_views 广播相乘
        attention_weights = attention_weights.view(1, self.num_views, 1)  # [1, num_views, 1]

        # 使用注意力权重对视图嵌入进行加权求和
        # [num_nodes, num_views, hidden_dim] * [1, num_views, 1] -> [num_nodes, num_views, hidden_dim]
        weighted_views = stacked_views * attention_weights
        fused_embedding = torch.sum(weighted_views, dim=1) # [num_nodes, hidden_dim]

        return fused_embedding, attention_weights.squeeze()

class NodeAttentionLayer(nn.Module):
    """
    节点级注意力层 (简化版)
    在 GCN 层之后，对邻居节点的聚合信息进行加权。
    这里我们使用一个简化的自注意力机制来重新加权 GCN 的输出。
    """
    def __init__(self, in_channels):
        super(NodeAttentionLayer, self).__init__()
        self.attention_layer = nn.Linear(in_channels, 1)

    def forward(self, x, edge_index):
        # 这是一个简化的实现，更复杂的实现可以使用 GATConv
        # 这里我们让每个节点基于其自身特征学习一个权重
        attention_scores = self.attention_layer(x)
        attention_weights = F.softmax(attention_scores, dim=0)
        
        # 将权重应用到节点特征上
        x_weighted = x * attention_weights
        return x_weighted

class SemiGNN(nn.Module):
    """
    多视图半监督图神经网络模型
    """
    def __init__(self, num_features, num_hidden, num_classes, num_views, view_dims):
        """
        Args:
            num_features (int): 输入总特征维度。
            num_hidden (int): 隐藏层维度。
            num_classes (int): 类别数 (合法/非法，即 2)。
            num_views (int): 视图数量。
            view_dims (list of int): 每个视图的特征维度。
        """
        super(SemiGNN, self).__init__()
        assert sum(view_dims) == num_features, "视图维度之和必须等于总特征维度"
        
        self.num_views = num_views
        self.view_dims = view_dims
        self.num_hidden = num_hidden

        # 为每个视图创建独立的 GCN 层
        self.view_gcns = nn.ModuleList()
        for dim in self.view_dims:
            self.view_gcns.append(GCNConv(dim, num_hidden))

        # 节点级注意力层
        self.node_attention = NodeAttentionLayer(num_hidden)

        # 视图级注意力层
        self.view_attention = ViewAttention(num_hidden, num_views)

        # 最终的分类器
        self.classifier = nn.Linear(num_hidden, num_classes)

    def forward(self, data):
        """
        前向传播

        Args:
            data (torch_geometric.data.Data): 图数据对象。

        Returns:
            tuple: (分类输出的 log_softmax, 融合后的节点嵌入, 视图注意力权重)
        """
        x, edge_index = data.x, data.edge_index

        # 1. 将输入特征按视图切分
        view_features = torch.split(x, self.view_dims, dim=1)
        
        # 2. 对每个视图分别进行图卷积
        view_embeddings = []
        for i in range(self.num_views):
            # GCN层
            view_emb = self.view_gcns[i](view_features[i], edge_index)
            view_emb = F.relu(view_emb)
            # 节点级注意力
            view_emb = self.node_attention(view_emb, edge_index)
            view_embeddings.append(view_emb)

        # 3. 视图级注意力融合
        fused_embedding, view_attention_weights = self.view_attention(view_embeddings)

        # 4. 分类
        output = self.classifier(fused_embedding)
        
        return F.log_softmax(output, dim=1), fused_embedding, view_attention_weights

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


if __name__ == '__main__':
    # --- 模拟数据进行模型测试 ---
    num_nodes = 100
    num_classes = 2 # 合法 vs 非法
    
    # 假设特征维度：交易(165), 传播(3), 社交(1)
    view_dims = [165, 3, 1]
    num_features = sum(view_dims)
    num_hidden = 64
    num_views = len(view_dims)

    # 创建模拟的 PyG Data 对象
    from torch_geometric.data import Data
    
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, 200), dtype=torch.long)
    y = torch.randint(0, num_classes, (num_nodes,), dtype=torch.long)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:70] = True # 假设前70个节点有标签
    
    mock_data = Data(x=x, edge_index=edge_index, y=y)
    mock_data.train_mask = train_mask

    # 初始化模型
    model = SemiGNN(
        num_features=num_features,
        num_hidden=num_hidden,
        num_classes=num_classes,
        num_views=num_views,
        view_dims=view_dims
    )
    print("模型结构:")
    print(model)

    # 前向传播测试
    output, embedding, view_weights = model(mock_data)
    print("\n--- 前向传播测试 ---")
    print("分类输出 shape:", output.shape)
    print("节点嵌入 shape:", embedding.shape)
    print("视图注意力权重:", view_weights)

    # 损失函数测试
    total_loss, cls_loss, rel_loss = loss_function(
        output, mock_data.y, embedding, mock_data.edge_index, mock_data.train_mask
    )
    print("\n--- 损失函数测试 ---")
    print(f"总损失: {total_loss.item():.4f}")
    print(f"分类损失: {cls_loss.item():.4f}")
    print(f"关系损失: {rel_loss.item():.4f}")
    
    # 检查模型参数是否在更新
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    initial_param = model.classifier.weight.clone()
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    updated_param = model.classifier.weight.clone()
    
    print("\n--- 参数更新测试 ---")
    assert not torch.equal(initial_param, updated_param), "模型参数没有更新！"
    print("模型参数已成功更新。")
