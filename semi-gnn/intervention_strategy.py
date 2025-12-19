# -*- coding: utf-8 -*-
"""
@File: intervention_strategy.py
@Time: 2025/11/23
@Description: 分级干预策略模拟器
本模块根据模型输出的诈骗风险概率，执行相应的分级干预策略。
主要功能包括：
1.  定义风险等级（高、中、低）。
2.  根据风险等级生成不同的干预措施文案。
3.  模拟接收用户反馈并预留模型微调接口的逻辑。
"""
import torch

def define_risk_levels(probabilities, thresholds):
    """
    根据概率和阈值，为每个节点定义风险等级。

    Args:
        probabilities (torch.Tensor or np.ndarray): 模型输出的诈骗概率（0-1范围）。
        thresholds (dict): 包含 'high' 和 'medium' 两个键的阈值字典。

    Returns:
        list: 每个节点对应的风险等级字符串 ('high', 'medium', 'low')。
    """
    risk_levels = []
    for prob in probabilities:
        if prob >= thresholds['high']:
            risk_levels.append('high')
        elif prob >= thresholds['medium']:
            risk_levels.append('medium')
        else:
            risk_levels.append('low')
    return risk_levels

def apply_intervention(node_id, risk_level):
    """
    根据单个节点的风险等级，生成相应的干预策略。

    Args:
        node_id (int): 节点的唯一标识符。
        risk_level (str): 节点的风险等级 ('high', 'medium', 'low')。

    Returns:
        dict: 包含干预动作和文案的字典。
    """
    strategy = {
        'node_id': node_id,
        'risk_level': risk_level,
        'actions': [],
        'message': ''
    }
    
    if risk_level == 'high':
        strategy['actions'] = ['auto_block_message', 'limit_account_functions', 'strong_warning']
        strategy['message'] = (
            f"[红色强提醒] 系统检测到与您交互的节点 {node_id} 存在极高诈骗风险！"
            "为保护您的安全，已自动屏蔽其信息，并临时限制其转账、添加好友等功能。"
            "请勿与对方发生任何资金往来！"
        )
    elif risk_level == 'medium':
        strategy['actions'] = ['yellow_warning_popup', 'log_subsequent_behavior']
        strategy['message'] = (
            f"[黄色预警] 系统检测到节点 {node_id} 存在较高诈骗风险。"
            "请谨慎互动，注意核实对方身份，切勿轻易转账或透露个人信息。"
            "您的后续交互将被记录以备核查。"
        )
    else: # low risk
        strategy['actions'] = ['push_educational_content']
        strategy['message'] = (
            f"节点 {node_id} 当前风险较低。温馨提示：网络环境复杂，"
            "时刻保持警惕是防范诈骗的最佳方式。点击查看《反诈科普手册》。"
        )
        
    return strategy

def simulate_intervention_workflow(model, data, node_indices_to_check):
    """
    对指定的一批节点，模拟完整的“预测-分级-干预”流程。

    Args:
        model (torch.nn.Module): 训练好的模型。
        data (torch_geometric.data.Data): 完整的图数据。
        node_indices_to_check (list or torch.Tensor): 需要检查的节点索引列表。

    Returns:
        list: 每个被检查节点的干预策略字典列表。
    """
    model.eval()
    with torch.no_grad():
        # 获取所有节点的诈骗概率
        output, _, _ = model(data)
        probabilities = torch.exp(output)[:, 1] # 非法节点的概率
    
    # 定义风险等级阈值
    thresholds = {'high': 0.90, 'medium': 0.60}
    
    # 获取指定节点的概率
    target_probs = probabilities[node_indices_to_check]
    
    # 定义风险等级
    risk_levels = define_risk_levels(target_probs, thresholds)
    
    # 应用干预策略
    intervention_results = []
    for i, node_idx in enumerate(node_indices_to_check):
        node_id = data.txId[node_idx].item() if hasattr(data, 'txId') else node_idx # 假设原始ID存在
        risk_level = risk_levels[i]
        strategy = apply_intervention(node_id, risk_level)
        intervention_results.append(strategy)
        
    return intervention_results

def feedback_loop_interface(feedback_data):
    """
    模拟处理用户反馈的接口。
    在实际应用中，此函数会将反馈数据格式化，并触发模型的微调（fine-tuning）流程。

    Args:
        feedback_data (list of dict): 用户反馈列表，每个字典包含 'node_id', 'feedback' ('valid' or 'misjudged'), 'true_label' (0 or 1)。
        
    Returns:
        dict: 包含需要更新的标签信息，用于后续的模型微调。
    """
    print("\n--- 接收到用户反馈，启动闭环模拟 ---")
    
    updates_for_finetuning = {
        'nodes_to_update': [],
        'new_labels': []
    }
    
    for feedback in feedback_data:
        node_id = feedback.get('node_id')
        judgement = feedback.get('feedback')
        true_label = feedback.get('true_label')
        
        if judgement == 'misjudged':
            print(f"收到对节点 {node_id} 的“误判”反馈。准备用于模型微调。")
            # 记录需要更新标签的节点及其正确标签
            updates_for_finetuning['nodes_to_update'].append(node_id)
            updates_for_finetuning['new_labels'].append(true_label)
        elif judgement == 'valid':
            print(f"收到对节点 {node_id} 的“提醒有效”反馈。此信息可用于增强模型置信度（此处未实现）。")
            
    if updates_for_finetuning['nodes_to_update']:
        print("\n模型微调接口已准备好以下数据：")
        print(f"待更新节点: {updates_for_finetuning['nodes_to_update']}")
        print(f"新 标 签: {updates_for_finetuning['new_labels']}")
        print("（在实际部署中，这些数据将被送入一个微调训练流程。）")
    
    return updates_for_finetuning


if __name__ == '__main__':
    # --- 模拟数据和模型进行测试 ---
    from model import SemiGNN
    from torch_geometric.data import Data

    # 模拟一个训练好的模型
    num_features = 185
    view_dims = [181, 3, 1]
    mock_model = SemiGNN(num_features=num_features, num_hidden=64, num_classes=2, num_views=3, view_dims=view_dims)
    
    # 模拟图数据
    num_nodes = 20
    x = torch.rand(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, 30), dtype=torch.long)
    # 假设原始 txId
    txId = torch.arange(1000, 1000 + num_nodes)
    mock_data = Data(x=x, edge_index=edge_index, txId=txId)

    # 模拟模型输出
    mock_model.eval()
    with torch.no_grad():
        mock_output, _, _ = mock_model(mock_data)
        # 手动设置一些概率以测试不同等级
        mock_probs = torch.exp(mock_output)[:, 1]
        mock_probs[0] = 0.95 # 高风险
        mock_probs[1] = 0.75 # 中风险
        mock_probs[2] = 0.45 # 低风险
        
        # 重写模型 forward 方法以返回预设概率
        def forward_mock(data):
            log_probs = torch.log(torch.stack([1 - mock_probs, mock_probs], dim=1))
            return log_probs, None, None
        mock_model.forward = forward_mock

    # --- 测试干预流程 ---
    nodes_to_check = [0, 1, 2, 3] # 检查前4个节点
    print("--- 开始模拟干预工作流 ---")
    interventions = simulate_intervention_workflow(mock_model, mock_data, nodes_to_check)
    
    print("\n干预策略结果：")
    for res in interventions:
        print(f"节点 {res['node_id']} (风险等级: {res['risk_level']}) -> 策略: {res['message']}")

    # --- 测试反馈闭环 ---
    mock_feedback = [
        {'node_id': 1000, 'feedback': 'valid'},
        {'node_id': 1001, 'feedback': 'misjudged', 'true_label': 0}, # 假设中风险是误判，应为合法
        {'node_id': 1002, 'feedback': 'valid'}
    ]
    
    finetuning_data = feedback_loop_interface(mock_feedback)
    
    print("\nIntervention Strategy 模块独立测试完成。")
