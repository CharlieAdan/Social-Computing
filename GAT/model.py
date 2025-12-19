# -*- coding: utf-8 -*-
"""
@File: gat_model.py
@Description: 图注意力网络 (GAT) 模型
用于对比实验，验证 GAT 是否能比 Semi-GNN 更好地捕捉欺诈特征。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_views=None, view_dims=None, heads=4, dropout=0.5):
        """
        初始化 GAT 模型。
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_classes: 类别数
            num_views: (兼容性参数，GAT不使用)
            view_dims: (兼容性参数，GAT不使用)
            heads: 多头注意力的头数
            dropout: Dropout 概率
        """
        super(GAT, self).__init__()
        self.dropout = dropout
        
        # 第一层 GAT: 输入 -> 隐藏层 (多头)
        # 输出维度将是 hidden_dim * heads
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        
        # 第二层 GAT: 隐藏层 -> 输出层 (单头，用于分类)
        self.conv2 = GATConv(hidden_dim * heads, num_classes, heads=1, concat=False, dropout=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Dropout on input features
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # First GAT layer
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        # 保存节点嵌入用于 Loss 计算 (Relation Consistency Loss)
        node_embeddings = x 
        
        # Dropout between layers
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GAT layer
        x = self.conv2(x, edge_index)
        
        # Log Softmax for classification
        # 返回三个值以保持与 Semi-GNN 接口一致: output, embedding, attention_weights(None)
        return F.log_softmax(x, dim=1), node_embeddings, None
