项目说明（按实验类型重构）

本仓库已将实验代码按方法拆分为两个独立文件夹：`semi-gnn`（Semi-GNN 实验）和 `GAT`（GAT 对比实验）。根目录保留共享数据与依赖清单。

重要路径：

- `data/elliptic/`           — 共享数据集（CSV 文件）。
- `temp_project/`            — 已整理的实验工程（可直接使用或解压至其他位置）。
  - `semi-gnn/`              — Semi-GNN 实验目录
    - `imageresult/`         — Semi-GNN 的输出（图、模型、PNG）
    - `main.py`, `model.py`, `trainer.py`, `loss.py`, `data_processor.py`, `intervention_strategy.py`
  - `GAT/`                   — GAT 对比实验目录
    - `imageresult/`         — GAT 的输出（图、模型、PNG）
    - `main.py`, `model.py`, `trainer.py`, `loss.py`, `data_processor.py`, `intervention_strategy.py`

- `requirements.txt`        — 依赖清单（放在根或 `temp_project/` 中均有拷贝）
- `project_code_v2.zip`     — 已整理并打包的项目归档（位于仓库根目录）

快速开始（Windows PowerShell）：

1) 运行 Semi-GNN 实验

```powershell
cd temp_project\semi-gnn
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install -r ..\requirements.txt
python main.py
```

2) 运行 GAT 实验

```powershell
cd temp_project\GAT
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install -r ..\requirements.txt
python main.py
```

运行结果与数据

- 各实验的模型与可视化输出会保存在对应目录下的 `imageresult/`。
- 请确保 `data/elliptic` 保持在仓库根目录（两个实验均用相对路径 `../data/elliptic` 访问）。

注意事项

- PyTorch Geometric (PyG) 需要根据你的 `torch`/CUDA 版本采用官网推荐的安装命令，请参照：https://pytorch-geometric.readthedocs.io
- 如果你要将整理后的 `temp_project` 推送到 GitHub：
  1. 在 GitHub 页面创建一个空仓库；
  2. 在本地执行：

```powershell
git remote add origin https://github.com/你的用户名/你的仓库名.git
git branch -M main
git push -u origin main
```

- 我在 `elegantbook.cls` 的封面中已插入占位的仓库地址；在你建立远端仓库后，可把真实 URL 替换进去以便封面显示代码链接。

如需我继续：
- 将 `temp_project` 的内容原地替换为仓库根的项目结构并修正所有 import 路径，或
- 帮你把本地结构推送到你指定的远端仓库（需要远端 URL 或本机 GitHub 授权）。

如需我继续执行其中任一操作，请确认要我执行哪一步。
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

