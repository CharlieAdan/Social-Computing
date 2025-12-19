# 基于多视图半监督图神经网络（Semi-GNN）的金融诈骗检测系统

项目说明

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
