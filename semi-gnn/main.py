# -*- coding: utf-8 -*-
"""
@File: main.py
@Time: 2025/11/23
@Description: 金融诈骗检测系统主入口 (Semi-GNN)
本文件是整个项目的核心调度器，负责串联并执行所有模块，实现从数据处理到模型训练、
评估，再到干预策略模拟的全流程自动化。

运行步骤：
1.  配置数据目录、模型超参数等。
2.  执行 `main()` 函数。
    -   `data_processor` 模块加载并处理数据。
    -   `model` 模块定义模型结构 (Semi-GNN)。
    -   `trainer` 模块负责模型的训练与评估，并保存最佳模型。
    -   `intervention_strategy` 模块加载训练好的模型，对指定节点进行风险预测并模拟干预。
    -   模拟接收用户反馈，并展示如何将反馈用于未来的模型迭代。
"""
import os
import torch
import numpy as np
import pandas as pd

# 导入自定义模块
from data_processor import load_and_process_data
from model import SemiGNN
from trainer import run_training
from intervention_strategy import simulate_intervention_workflow, feedback_loop_interface

def setup_environment(data_dir):
    """
    准备运行环境，包括创建模拟数据（如果不存在）。
    """
    print("--- 1. 环境设置与数据准备 ---")
    if not os.path.exists(data_dir):
        print(f"数据目录 '{data_dir}' 不存在，将创建并生成模拟数据。")
        os.makedirs(data_dir)
        # 创建模拟的CSV文件以确保代码可运行
        num_nodes, num_edges = 2000, 5000
        num_labeled = 800
        
        mock_features = pd.DataFrame({
            'txId': range(num_nodes),
            'Time step': np.random.randint(1, 50, num_nodes),
            **{f'feature_{i}': np.random.rand(num_nodes) for i in range(181)}
        })
        mock_edges = pd.DataFrame({
            'txId1': np.random.randint(0, num_nodes, num_edges),
            'txId2': np.random.randint(0, num_nodes, num_edges)
        })
        mock_classes = pd.DataFrame({
            'txId': range(num_labeled),
            'class': np.random.choice(['1', '2', 'unknown'], num_labeled, p=[0.45, 0.1, 0.45])
        })
        
        mock_features.to_csv(f"{data_dir}/elliptic_txs_features.csv", index=False)
        mock_edges.to_csv(f"{data_dir}/elliptic_txs_edgelist.csv", index=False)
        mock_classes.to_csv(f"{data_dir}/elliptic_txs_classes.csv", index=False)
        print("模拟数据文件已生成。")
    else:
        print(f"使用现有数据目录: '{data_dir}'")

def main():
    """
    主函数，执行金融诈骗检测全流程。
    """
    # --- 配置参数 ---
    CONFIG = {
        'data_dir': '../data/elliptic',
        'output_dir': 'imageresult', # Semi-GNN 策略结果目录
        'model_params': {
            'hidden_dim': 128,
            'num_classes': 2, # 合法 vs 非法
            'num_views': 3,
            # 视图维度将在数据加载后动态确定
            'view_dims': None 
        },
        'training_params': {
            'learning_rate': 1e-3,
            'weight_decay': 5e-4,
            'epochs': 50, # 增加 epoch 数量以获得更好效果
            'reg_lambda': 0.01, # 关系损失的权重
            'use_class_weights': True, # 启用类别权重
            'custom_class_weights': [1.0, 3.0], # 手动设置权重：非法交易(1)权重为3，合法(0)为1
            'use_focal': False # 关闭 Focal Loss，回归经典加权
        }
    }

    # --- 1. 环境与数据准备 ---
    setup_environment(CONFIG['data_dir'])
    
    # --- 2. 数据加载与预处理 ---
    print("\n--- 2. 开始数据加载与预处理 ---")
    try:
        graph_data = load_and_process_data(CONFIG['data_dir'])
        # 动态确定视图维度
        # 假设：前181个是交易特征，接下来3个是传播特征，最后1个是社交特征
        # 这个需要根据 data_processor.py 的实现来调整
        num_tx_features = 181 
        num_prop_features = 3
        num_social_features = 1
        
        # 实际特征数量可能因数据处理而变化，进行动态调整
        total_features = graph_data.x.shape[1]
        other_features = total_features - num_prop_features - num_social_features
        view_dims = [other_features, num_prop_features, num_social_features]
        CONFIG['model_params']['view_dims'] = view_dims
        
        print(f"动态确定的视图维度: {view_dims}")

    except Exception as e:
        print(f"数据处理失败: {e}")
        return

    # --- 3. 模型构建 ---
    print("\n--- 3. 构建 Semi-GNN 模型 ---")
    model = SemiGNN(
        num_features=graph_data.num_features,
        num_hidden=CONFIG['model_params']['hidden_dim'],
        num_classes=CONFIG['model_params']['num_classes'],
        num_views=CONFIG['model_params']['num_views'],
        view_dims=CONFIG['model_params']['view_dims']
    )
    print("模型结构:")
    print(model)

    # --- 4. 模型训练与评估 ---
    print("\n--- 4. 开始模型训练与评估 ---")
    training_config = {
        **CONFIG['training_params'],
        'output_dir': CONFIG['output_dir']
    }
    trained_model = run_training(graph_data, model, training_config)
    
    # --- 5. 加载最佳模型并进行干预策略模拟 ---
    print("\n--- 5. 加载最佳模型并模拟干预策略 ---")
    best_model_path = os.path.join(CONFIG['output_dir'], 'best_model.pth')
    if os.path.exists(best_model_path):
        # 加载模型权重
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trained_model.load_state_dict(torch.load(best_model_path, map_location=device))
        trained_model.to(device)
        graph_data.to(device)
        print("最佳模型已加载。")

        # 随机选择一些节点进行干预模拟
        num_nodes_to_check = 10
        # 确保选择的节点在图范围内
        indices_to_check = torch.randperm(graph_data.num_nodes)[:num_nodes_to_check]
        
        # 为了演示效果，手动加入测试集中的已知非法节点（如果存在）
        illicit_nodes = torch.where((graph_data.y == 1) & (graph_data.test_mask))[0]
        if len(illicit_nodes) > 0:
            indices_to_check = torch.cat([indices_to_check, illicit_nodes[:min(3, len(illicit_nodes))]])

        print(f"将对 {len(indices_to_check)} 个节点进行干预模拟...")
        
        # 添加原始ID到data对象，以便在干预模块中使用
        # 考虑到文件可能没有表头，直接读取第一列
        try:
            node_ids_df = pd.read_csv(f"{CONFIG['data_dir']}/elliptic_txs_features.csv", usecols=['txId'])
        except ValueError:
            # 如果报错说明没有表头，读取第一列
            node_ids_df = pd.read_csv(f"{CONFIG['data_dir']}/elliptic_txs_features.csv", header=None, usecols=[0])
            node_ids_df.columns = ['txId']
            
        id_map = {i: txid for i, txid in enumerate(node_ids_df['txId'])}
        graph_data.txId = [id_map.get(i, -1) for i in range(graph_data.num_nodes)]
        graph_data.txId = torch.tensor(graph_data.txId)


        intervention_results = simulate_intervention_workflow(trained_model, graph_data, indices_to_check)
        
        print("\n干预模拟结果:")
        for result in intervention_results:
            print(f"  - 节点 {result['node_id']}: 风险等级 '{result['risk_level']}', 触发动作: {result['actions']}")
            # print(f"    消息: {result['message']}")
    else:
        print("未找到已训练的模型，跳过干预模拟。")

    # --- 6. 模拟反馈闭环 ---
    print("\n--- 6. 模拟用户反馈与闭环学习 ---")
    # 假设我们从干预结果中收到了一些反馈
    if 'intervention_results' in locals() and intervention_results:
        mock_feedback = []
        for result in intervention_results[:3]: # 取前三个结果模拟反馈
            node_id = result['node_id']
            # 模拟一个误判和一个有效提醒
            if result['risk_level'] == 'high':
                mock_feedback.append({'node_id': node_id, 'feedback': 'valid'})
            elif result['risk_level'] == 'medium':
                # 假设这个中风险是误判，它其实是合法的
                mock_feedback.append({'node_id': node_id, 'feedback': 'misjudged', 'true_label': 0})
            else:
                 mock_feedback.append({'node_id': node_id, 'feedback': 'valid'})
        
        feedback_loop_interface(mock_feedback)
    else:
        print("无干预结果，跳过反馈模拟。")
        
    print("\n\n金融诈骗检测全流程执行完毕！")


if __name__ == '__main__':
    main()
