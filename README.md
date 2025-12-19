# 基于多视图半监督图神经网络（Semi-GNN）的金融诈骗检测系统

本项目实现了一个用于金融交易与社交平台诈骗检测的端到端系统，包含数据预处理、multi-view 半监督 GNN 模型、训练/评估与分级干预模拟。

目录结构
- `data_processor.py`：数据加载、清洗与特征工程（交易、传播、社交三类特征）。
- `semi_gnn_model.py`：Semi-GNN 模型实现（节点级 + 视图级注意力，双重损失）。
- `trainer.py`：训练、评估、可视化与模型保存。
- `intervention_strategy.py`：分级干预逻辑与反馈闭环接口。
- `main.py`：主入口，串联完整流程（数据→训练→评估→干预）。
- `requirements.txt`：依赖包列表与安装说明。

运行环境
- Python 3.8+
- 推荐使用虚拟环境（venv 或 conda）

依赖安装
1. 创建并激活虚拟环境（PowerShell 示例）：
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
```
2. 安装核心依赖（注意：PyG 需按 PyTorch 与 CUDA 版本选择安装命令）
```powershell
pip install -r requirements.txt
# PyG 建议按照官网指令安装相关扩展：
# https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
```

快速运行（使用真实 Elliptic 数据）
1. 将 Elliptic 三个 CSV 文件放入 `data/elliptic/`：
   - `elliptic_txs_features.csv`
   - `elliptic_txs_edgelist.csv`
   - `elliptic_txs_classes.csv`
2. 运行主脚本（默认会执行完整训练流程）：
```powershell
python main.py
```
3. 训练输出与可视化文件将保存在 `results/`（或短跑时在 `short_run_results/`）。

说明与注意事项
- `data_processor.py` 会自动检测 CSV 是否包含表头；若数据没有表头，会尝试生成默认列名（`txId`, `Time step`, `feature_...`）。
- 由于 PyTorch Geometric 依赖于特定的二进制包（如 `torch-scatter`、`torch-sparse` 等），请根据你的 PyTorch 版本与操作系统，参考 PyG 官方安装说明来安装对应轮次的扩展包。
- `trainer.py` 会在训练过程中保存最佳模型到 `results/best_model.pth`（以测试集 F1-score 为准）。

交付物
- 源代码：本仓库中的 `.py` 文件
- 训练产物（示例短训练）：`short_run_results/best_model.pth`, `loss_curve.png`, `roc_curve.png`, `risk_heatmap.png`

