# -*- coding: utf-8 -*-
"""
@File: trainer.py
@Time: 2025/11/23
@Description: 模型训练与评估器
本模块负责 Semi-GNN 模型的训练、评估和保存。
主要功能包括：
1.  完整的训练流程，包括前向传播、损失计算、反向传播和参数优化。
2.  在测试集上进行模型评估，计算准确率、精确率、召回率、F1-score 和 ROC-AUC。
3.  保存性能最优的模型权重。
4.  可视化训练过程，如损失曲线、ROC 曲线和节点风险热力图。
"""
import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from model import SemiGNN
from loss import loss_function
from torch_geometric.utils import to_dense_adj

def train_one_epoch(model, data, optimizer, reg_lambda, class_weights=None, use_focal=False):
    """
    执行一个 epoch 的训练。
    """
    model.train()
    optimizer.zero_grad()
    
    output, embedding, _ = model(data)
    
    total_loss, cls_loss, rel_loss = loss_function(
        output, data.y, embedding, data.edge_index, data.train_mask, reg_lambda, class_weights, use_focal
    )
    
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item(), cls_loss.item(), rel_loss.item()

@torch.no_grad()
def evaluate(model, data, mask):
    """
    在指定数据集（由 mask 定义）上评估模型。
    """
    model.eval()
    output, _, _ = model(data)
    
    # 获取预测结果
    pred_prob = torch.exp(output[mask])[:, 1] # 非法节点的概率
    
    true_labels = data.y[mask]
    
    # 寻找最佳阈值 (仅在评估模式下)
    best_f1 = 0
    best_threshold = 0.5
    
    # 简单的阈值搜索
    thresholds = np.arange(0.1, 0.9, 0.05)
    for th in thresholds:
        temp_pred = (pred_prob >= th).long()
        temp_f1 = f1_score(true_labels.cpu(), temp_pred.cpu(), zero_division=0)
        if temp_f1 > best_f1:
            best_f1 = temp_f1
            best_threshold = th
            
    # 使用最佳阈值进行最终预测
    pred_class = (pred_prob >= best_threshold).long()
    
    # 计算各项指标
    accuracy = accuracy_score(true_labels.cpu(), pred_class.cpu())
    precision = precision_score(true_labels.cpu(), pred_class.cpu(), zero_division=0)
    recall = recall_score(true_labels.cpu(), pred_class.cpu(), zero_division=0)
    f1 = f1_score(true_labels.cpu(), pred_class.cpu(), zero_division=0)
    try:
        roc_auc = roc_auc_score(true_labels.cpu(), pred_prob.cpu())
    except ValueError:
        roc_auc = 0.5 # 如果只有一个类别，无法计算AUC

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'best_threshold': best_threshold
    }
    
    return metrics, pred_prob.cpu(), true_labels.cpu()

def plot_loss_curves(train_losses, test_losses, save_path):
    """绘制训练和测试损失曲线。"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    # plt.plot(test_losses, label='Test Loss') # 评估时通常不计算 test loss
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'loss_curve.png'))
    plt.close()

def plot_roc_curve(true_labels, pred_probs, save_path):
    """绘制 ROC 曲线。"""
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    roc_auc = roc_auc_score(true_labels, pred_probs)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'roc_curve.png'))
    plt.close()

def plot_risk_heatmap(model, data, save_path, num_nodes_to_show=100):
    """
    绘制节点风险热力图（简化版）。
    由于节点太多，只显示部分高风险节点及其一阶邻居。
    """
    # 获取所有节点的风险概率（通过模型预测得到完整概率向量）
    model.eval()
    with torch.no_grad():
        full_output, _, _ = model(data)
    full_probs = torch.exp(full_output)[:, 1].cpu().numpy()

    high_risk_threshold = 0.9
    high_risk_indices = np.where(full_probs >= high_risk_threshold)[0]

    if len(high_risk_indices) == 0:
        print("未发现高风险节点，不生成风险热力图。")
        return

    # 选择部分高风险节点进行可视化
    nodes_to_visualize = set(high_risk_indices[:num_nodes_to_show])
    
    # 添加其一阶邻居
    edge_index_np = data.edge_index.cpu().numpy()
    neighbors = set()
    for node_idx in nodes_to_visualize:
        # 入边邻居
        connected_nodes_in = edge_index_np[0, edge_index_np[1] == node_idx]
        # 出边邻居
        connected_nodes_out = edge_index_np[1, edge_index_np[0] == node_idx]
        neighbors.update(connected_nodes_in)
        neighbors.update(connected_nodes_out)
    
    nodes_to_visualize.update(neighbors)
    nodes_to_visualize = list(nodes_to_visualize)
    
    if len(nodes_to_visualize) > 500: # 限制最大节点数防止图像过大
        nodes_to_visualize = nodes_to_visualize[:500]

    subgraph_adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0][nodes_to_visualize, :][:, nodes_to_visualize]
    
    plt.figure(figsize=(12, 12))
    sns.heatmap(subgraph_adj.cpu().numpy(), cmap="YlGnBu", xticklabels=False, yticklabels=False)
    plt.title(f'Risk Adjacency Heatmap (showing {len(nodes_to_visualize)} nodes)')
    plt.xlabel('Nodes')
    plt.ylabel('Nodes')
    plt.savefig(os.path.join(save_path, 'risk_heatmap.png'))
    plt.close()


def run_training(data, model, config):
    """
    完整的训练和评估流程。

    Args:
        data (torch_geometric.data.Data): 图数据。
        model (torch.nn.Module): 要训练的模型。
        config (dict): 包含超参数的配置字典。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = model.to(device)
    data = data.to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config.get('weight_decay', 5e-4)
    )

    best_f1_score = 0.0
    train_losses, test_metrics_history = [], []
    
    output_dir = config.get('output_dir', 'results')
    os.makedirs(output_dir, exist_ok=True)

    # 计算类别权重
    class_weights = None
    if config.get('use_class_weights', False):
        # 检查是否提供了自定义权重
        if 'custom_class_weights' in config:
            class_weights = torch.tensor(config['custom_class_weights']).float().to(device)
            print(f"使用自定义类别权重: {class_weights.tolist()}")
        else:
            # 只根据训练集计算权重
            train_labels = data.y[data.train_mask]
            class_counts = torch.bincount(train_labels)
            # 避免除以零
            class_counts = class_counts.float() + 1e-6
            total_samples = len(train_labels)
            num_classes = len(class_counts)
            
            # 权重计算公式: total / (num_classes * count)
            class_weights = total_samples / (num_classes * class_counts)
            class_weights = class_weights.to(device)
            print(f"已启用自动类别权重平衡: {class_weights.tolist()}")

    print("开始训练...")
    use_focal = config.get('use_focal', False)
    if use_focal:
        print("已启用 Focal Loss。")

    for epoch in range(config['epochs']):
        total_loss, cls_loss, rel_loss = train_one_epoch(
            model, data, optimizer, config['reg_lambda'], class_weights, use_focal
        )
        
        train_losses.append(total_loss)
        
        # 在测试集上评估
        test_metrics, test_probs, test_labels = evaluate(model, data, data.test_mask)
        test_metrics_history.append(test_metrics)
        
        print(
            f"Epoch {epoch+1:03d} | "
            f"Total Loss: {total_loss:.4f} (CLS: {cls_loss:.4f}, REL: {rel_loss:.4f}) | "
            f"Test F1: {test_metrics['f1_score']:.4f} | "
            f"Test AUC: {test_metrics['roc_auc']:.4f}"
        )

        # 保存最佳模型
        if test_metrics['f1_score'] > best_f1_score:
            best_f1_score = test_metrics['f1_score']
            model_path = os.path.join(output_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"发现更优模型，F1-score: {best_f1_score:.4f}，已保存至 {model_path}")

    print("训练完成。")
    
    # --- 可视化 ---
    print("开始生成可视化图表...")
    plot_loss_curves(train_losses, [], output_dir)
    
    # 使用最佳模型进行最终评估和可视化
    best_model_path = os.path.join(output_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print("已加载性能最佳的模型进行最终评估。")
    
    final_metrics, final_probs, final_labels = evaluate(model, data, data.test_mask)
    print("\n--- 最终测试集评估结果 ---")
    for key, value in final_metrics.items():
        print(f"{key.capitalize()}: {value:.4f}")
        
    plot_roc_curve(final_labels, final_probs, output_dir)
    
    # 绘制风险热力图（注意：这可能非常耗时和消耗内存）
    try:
        plot_risk_heatmap(model, data, output_dir)
    except Exception as e:
        print(f"生成风险热力图失败: {e}")
        
    print(f"所有可视化结果已保存至目录: {output_dir}")
    return model


if __name__ == '__main__':
    # --- 模拟数据和配置进行测试 ---
    from torch_geometric.data import Data
    from data_processor import load_and_process_data # 假设可以从 data_processor 导入

    # 模拟配置
    config = {
        'learning_rate': 1e-3,
        'weight_decay': 5e-4,
        'epochs': 10, # 仅用于测试
        'reg_lambda': 0.01,
        'hidden_dim': 64,
        'num_layers': 2, # GCN层数，模型中已固定，此处为参考
        'output_dir': 'test_results'
    }

    # 模拟数据
    # 尝试加载真实数据结构的模拟数据
    DATA_DIR = 'data/elliptic'
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        # 创建模拟文件
        pd.DataFrame({
            'txId': range(200), 'Time step': np.random.randint(1, 5, 200),
            **{f'feature_{i}': np.random.rand(200) for i in range(181)}
        }).to_csv(f"{DATA_DIR}/elliptic_txs_features.csv", index=False)
        pd.DataFrame({
            'txId1': np.random.randint(0, 200, 400), 'txId2': np.random.randint(0, 200, 400)
        }).to_csv(f"{DATA_DIR}/elliptic_txs_edgelist.csv", index=False)
        pd.DataFrame({
            'txId': range(150), 'class': np.random.choice(['1', '2', 'unknown'], 150, p=[0.6, 0.2, 0.2])
        }).to_csv(f"{DATA_DIR}/elliptic_txs_classes.csv", index=False)

    # 假设 view_dims 是从 data_processor 获取的
    # 交易特征: 183-2=181, 传播特征: 3, 社交特征: 1
    view_dims = [181, 3, 1] 
    num_features = sum(view_dims)

    # 加载数据
    mock_data = load_and_process_data(DATA_DIR)
    
    # 修正特征维度
    num_features_actual = mock_data.x.shape[1]
    # 假设多出来的特征都属于交易特征
    view_dims = [num_features_actual - 3 - 1, 3, 1]

    # 初始化模型
    model = SemiGNN(
        num_features=num_features_actual,
        num_hidden=config['hidden_dim'],
        num_classes=2, # illicit vs licit
        num_views=len(view_dims),
        view_dims=view_dims
    )

    # 运行训练流程
    run_training(mock_data, model, config)
    print("\nTrainer 模块独立测试完成。")
