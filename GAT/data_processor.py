# -*- coding: utf-8 -*-
"""
@File: data_processor.py
@Time: 2025/11/23
@Description: 数据处理器
本模块负责金融诈骗检测项目的数据加载、预处理、特征工程和图构建。
主要功能包括：
1.  加载 Elliptic++ 数据集的交易、边和类别数据。
2.  数据清洗，处理缺失值和异常值。
3.  特征工程，提取交易、传播和社交三类特征。
4.  构建 PyTorch Geometric (PyG) 所需的图数据结构。

运行步骤：
1.  确保 Elliptic++ 数据集文件（elliptic_txs_features.csv, elliptic_txs_edgelist.csv, elliptic_txs_classes.csv）位于指定的数据目录下。
2.  调用 `load_and_process_data(data_dir)` 函数，返回处理好的 PyG Data 对象。
"""
import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

def load_elliptic_data(data_dir: str):
    """
    加载 Elliptic++ 数据集的三个核心文件。

    Args:
        data_dir (str): 存放数据集文件的目录路径。

    Returns:
        tuple: 包含交易特征、边列表、节点类别三个 DataFrame。
    """
    print("开始加载 Elliptic++ 数据集...")
    try:
        # 为了兼容用户提供的 CSV 可能没有表头的情况，先读取首行判断是否包含列名
        def _read_with_header_auto(path, default_names_fn):
            # 使用文本模式逐行读取首行，避免分片读取导致首行不完整
            try:
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    first_line = f.readline().strip()
            except Exception:
                with open(path, 'r', encoding='latin1', errors='replace') as f:
                    first_line = f.readline().strip()

            # 如果首行包含字母（如 'txId'），则认为有表头
            header_present = any(c.isalpha() for c in first_line)
            if header_present:
                return pd.read_csv(path)
            else:
                # 没有表头，需要根据列数生成默认列名（使用 csv 分隔规则更稳健）
                import csv
                with open(path, 'r', encoding='utf-8', errors='replace', newline='') as f:
                    reader = csv.reader(f)
                    first_row = next(reader)
                ncols = len(first_row)
                names = default_names_fn(ncols)
                return pd.read_csv(path, header=None, names=names)

        features_path = f"{data_dir}/elliptic_txs_features.csv"
        edges_path = f"{data_dir}/elliptic_txs_edgelist.csv"
        classes_path = f"{data_dir}/elliptic_txs_classes.csv"

        features_df = _read_with_header_auto(
            features_path,
            lambda n: ['txId', 'Time step'] + [f'feature_{i}' for i in range(n-2)]
        )

        edges_df = _read_with_header_auto(
            edges_path,
            lambda n: ['txId1', 'txId2'] + [f'col_{i}' for i in range(2, n)]
        )

        classes_df = _read_with_header_auto(
            classes_path,
            lambda n: ['txId', 'class'] + [f'col_{i}' for i in range(2, n)]
        )

        # 统一 ID 字段的数据类型，避免 merge/map 时类型不匹配
        try:
            features_df['txId'] = features_df['txId'].astype(int)
        except Exception:
            features_df['txId'] = pd.to_numeric(features_df['txId'], errors='coerce').astype('Int64')

        if 'txId1' in edges_df.columns and 'txId2' in edges_df.columns:
            edges_df['txId1'] = edges_df['txId1'].astype(int)
            edges_df['txId2'] = edges_df['txId2'].astype(int)

        try:
            classes_df['txId'] = classes_df['txId'].astype(int)
        except Exception:
            classes_df['txId'] = pd.to_numeric(classes_df['txId'], errors='coerce').astype('Int64')

        print("数据集加载成功。")
        return features_df, edges_df, classes_df
    except FileNotFoundError as e:
        print(f"错误：找不到数据集文件，请检查路径 {data_dir}。")
        raise e

def clean_data(features_df: pd.DataFrame):
    """
    数据清洗：处理缺失值和异常值。
    对于 Elliptic 数据集，特征是匿名的，这里仅作示例性处理。
    实际应用中应根据特征的具体含义制定更精细的策略。

    Args:
        features_df (pd.DataFrame): 原始节点特征 DataFrame。

    Returns:
        pd.DataFrame: 清洗后的特征 DataFrame。
    """
    print("开始数据清洗...")
    # 简单填充缺失值（例如，用均值或中位数）
    # 在此数据集中，没有明确的缺失值，此步骤为通用流程展示
    if features_df.isnull().sum().sum() > 0:
        features_df = features_df.fillna(features_df.mean())
        print("缺失值已使用均值填充。")

    # 异常值处理（例如，使用 Z-score 或 IQR，此处未作处理）
    # ...

    print("数据清洗完成。")
    return features_df

def extract_features(features_df: pd.DataFrame, edges_df: pd.DataFrame):
    """
    特征工程：提取交易、传播和社交三类特征。

    Args:
        features_df (pd.DataFrame): 包含原始特征的 DataFrame。
        edges_df (pd.DataFrame): 边列表 DataFrame。

    Returns:
        pd.DataFrame: 包含所有特征的 DataFrame。
    """
    print("开始特征工程...")
    # 1. 交易特征 (Transaction Features)
    # 原始特征已包含，此处主要是标准化
    # 强制将特征列转换为数值类型并填充缺失值，避免 StandardScaler 报错
    feature_cols = features_df.columns.tolist()[2:]
    features_df[feature_cols] = features_df[feature_cols].apply(lambda col: pd.to_numeric(col, errors='coerce'))
    # 用列均值填充可能存在的 NaN
    for col in feature_cols:
        if features_df[col].isnull().any():
            mean_val = features_df[col].mean()
            features_df[col] = features_df[col].fillna(mean_val)

    tx_features = features_df.iloc[:, 2:].values
    scaler = StandardScaler()
    tx_features_scaled = scaler.fit_transform(tx_features)
    features_df.iloc[:, 2:] = tx_features_scaled
    print("交易特征已标准化。")

    # 创建图以计算网络特征
    G = nx.from_pandas_edgelist(edges_df, "txId1", "txId2", create_using=nx.DiGraph())

    # 2. 传播特征 (Propagation Features) - 基于SNA
    print("计算传播特征（中心度）...")
    # 计算度中心性（in-degree, out-degree）
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    
    # 将中心度特征加入DataFrame，确保 txId 的类型一致
    features_df['in_degree'] = features_df['txId'].map(lambda x: in_degree.get(int(x), 0) if not pd.isna(x) else 0)
    features_df['out_degree'] = features_df['txId'].map(lambda x: out_degree.get(int(x), 0) if not pd.isna(x) else 0)

    # PageRank
    print("计算 PageRank...")
    pagerank = nx.pagerank(G, alpha=0.85)
    features_df['pagerank'] = features_df['txId'].map(lambda x: pagerank.get(int(x), 0) if not pd.isna(x) else 0)
    print("传播特征提取完成。")

    # 3. 社交特征 (Social Features) - 模拟
    # 模拟用户互动强度，这里简化为基于共同邻居的权重
    # 实际场景中可能需要更复杂的模型，如 NLP 分析文本、用户画像等
    print("模拟社交特征...")
    # 示例：计算Jaccard相似系数作为社交互动强度的代理
    # 由于计算所有节点对的Jaccard系数非常耗时，此处仅为逻辑示意
    # 在实际应用中，可以针对特定节点或子图进行计算
    # 此处我们仅添加一个占位符特征
    features_df['social_interaction_mock'] = np.random.rand(len(features_df))
    print("社交特征模拟完成。")

    # 标准化新增的特征
    new_feature_cols = ['in_degree', 'out_degree', 'pagerank', 'social_interaction_mock']
    # 若这些列存在 NaN，先用均值填充，然后标准化
    for col in new_feature_cols:
        if features_df[col].isnull().any():
            features_df[col] = features_df[col].fillna(features_df[col].mean())
    features_df[new_feature_cols] = scaler.fit_transform(features_df[new_feature_cols])
    
    print("特征工程完成。")
    return features_df

def build_graph_data(features_df: pd.DataFrame, edges_df: pd.DataFrame, classes_df: pd.DataFrame):
    """
    构建 PyTorch Geometric 的 Data 对象。

    Args:
        features_df (pd.DataFrame): 包含所有特征的 DataFrame。
        edges_df (pd.DataFrame): 边列表 DataFrame。
        classes_df (pd.DataFrame): 节点类别 DataFrame。

    Returns:
        torch_geometric.data.Data: 构建好的图数据对象。
    """
    print("开始构建图数据结构...")
    
    # 合并类别信息
    # class 'unknown' -> -1, '1' (illicit) -> 1, '2' (licit) -> 0
    # 注意：Elliptic数据集原始标签中 '1' 代表非法(Illicit)，'2' 代表合法(Licit)
    # 我们将非法交易设为正样本(1)，合法交易设为负样本(0)
    class_map = {'unknown': -1, '1': 1, '2': 0}
    classes_df['class'] = classes_df['class'].map(class_map)
    
    # 将类别信息与特征合并
    full_df = pd.merge(features_df, classes_df, on='txId', how='left')
    full_df['class'] = full_df['class'].fillna(-1) # 未在 class 文件中出现的节点也视为未知

    # 节点ID到索引的映射
    node_ids = full_df['txId'].values
    node_id_map = {node_id: i for i, node_id in enumerate(node_ids)}

    # 节点特征矩阵
    feature_cols = [col for col in full_df.columns if col not in ['txId', 'Time step', 'class']]
    x = torch.tensor(full_df[feature_cols].values, dtype=torch.float)

    # 标签向量
    y = torch.tensor(full_df['class'].values, dtype=torch.long)

    # 边索引和边权重
    # 根据节点ID映射转换边
    edges_df['txId1_idx'] = edges_df['txId1'].map(node_id_map)
    edges_df['txId2_idx'] = edges_df['txId2'].map(node_id_map)
    
    # 过滤掉不在映射中的边
    valid_edges = edges_df.dropna(subset=['txId1_idx', 'txId2_idx'])
    
    edge_index = torch.tensor(valid_edges[['txId1_idx', 'txId2_idx']].values, dtype=torch.long).t().contiguous()

    # 模拟边权重（例如，基于交易金额或时间差，此处简化为1）
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)
    
    # 创建 Data 对象
    data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_weight)

    # 创建训练、验证、测试集的掩码
    # Elliptic 数据集按时间步划分，1-34为训练，35-49为测试
    # 这里我们遵循半监督设置，使用所有已知标签进行训练和评估
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    
    known_indices = torch.where(data.y != -1)[0]
    
    # 按7:3划分已知标签数据
    perm = torch.randperm(known_indices.size(0))
    split_point = int(0.7 * known_indices.size(0))
    
    train_indices = known_indices[perm[:split_point]]
    test_indices = known_indices[perm[split_point:]]
    
    train_mask[train_indices] = True
    test_mask[test_indices] = True
    
    data.train_mask = train_mask
    data.test_mask = test_mask
    
    print("图数据结构构建完成。")
    print(f"图信息概览:\n{data}")
    print(f"训练集节点数: {data.train_mask.sum().item()}")
    print(f"测试集节点数: {data.test_mask.sum().item()}")
    print(f"未知标签节点数: {data.num_nodes - data.train_mask.sum().item() - data.test_mask.sum().item()}")
    
    return data

def load_and_process_data(data_dir: str):
    """
    主函数，串联整个数据处理流程。

    Args:
        data_dir (str): 数据集目录。

    Returns:
        torch_geometric.data.Data: 最终的图数据对象。
    """
    features_df, edges_df, classes_df = load_elliptic_data(data_dir)
    features_df = clean_data(features_df)
    features_df = extract_features(features_df, edges_df)
    graph_data = build_graph_data(features_df, edges_df, classes_df)
    return graph_data

if __name__ == '__main__':
    # 使用示例
    # 假设数据集在 'data/elliptic' 目录下
    DATA_DIR = 'data/elliptic' 
    
    # 为了能独立运行此脚本，需要模拟数据目录和文件
    import os
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"创建模拟数据目录: {DATA_DIR}")
        # 创建模拟的CSV文件
        mock_features = pd.DataFrame({
            'txId': range(100),
            'Time step': np.random.randint(1, 50, 100),
            **{f'feature_{i}': np.random.rand(100) for i in range(181)}
        })
        mock_edges = pd.DataFrame({
            'txId1': np.random.randint(0, 100, 200),
            'txId2': np.random.randint(0, 100, 200)
        })
        mock_classes = pd.DataFrame({
            'txId': range(50), # 只有部分节点有标签
            'class': np.random.choice(['1', '2', 'unknown'], 50, p=[0.4, 0.1, 0.5])
        })
        mock_features.to_csv(f"{DATA_DIR}/elliptic_txs_features.csv", index=False)
        mock_edges.to_csv(f"{DATA_DIR}/elliptic_txs_edgelist.csv", index=False)
        mock_classes.to_csv(f"{DATA_DIR}/elliptic_txs_classes.csv", index=False)
        print("已生成模拟数据文件。")

    try:
        data = load_and_process_data(DATA_DIR)
        print("\n数据处理流程成功完成！")
        print("最终的 PyG Data 对象:")
        print(data)
    except Exception as e:
        print(f"\n数据处理过程中发生错误: {e}")
