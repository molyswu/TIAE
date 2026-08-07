<div align="center">
  <h1>TIAE：拓扑意图-动作嵌入</h1>
  <h3>面向目标导向具身控制的拓扑理论</h3>

  <p><strong>学习从意图流形到动作流形的拓扑保持映射 — 无需对齐，无需搜索，只需结构。</strong></p>

  <p>
    Yuxiang Wu<sup>1*</sup> &nbsp;&middot;&nbsp;
    Tusun Wu<sup>2*</sup> &nbsp;&middot;&nbsp;
    Yuyan Wu<sup>3</sup>
  </p>
  <p>
    <sup>1</sup>香港教育大学 &nbsp; <sup>2</sup>深圳迈特芯科技 &nbsp; <sup>3</sup>广东技术师范大学<br>
    <sup>*</sup>通讯作者
  </p>
  <p>
    <a href="#概述">概述</a> &nbsp;&middot;&nbsp;
    <a href="#安装">安装</a> &nbsp;&middot;&nbsp;
    <a href="#训练">训练</a> &nbsp;&middot;&nbsp;
    <a href="#评估">评估</a> &nbsp;&middot;&nbsp;
    <a href="#结果">结果</a> &nbsp;&middot;&nbsp;
    <a href="README.md">English</a>
  </p>
</div>

---

## 概述

**TIAE（Topological Intent-Action Embedding，拓扑意图-动作嵌入）**
是一个基于 JEPA 的世界模型框架，通过学习保持意图流形与动作流形之间的
**拓扑结构**来实现目标导向控制。TIAE 不强制特定意图对应特定动作
（"对齐假设"），而是保持成对距离结构 — 相似的意图诱导出相似的动作分布，
远离的意图映射到不同的动作律。

### 拓扑与对齐的比较

| 对齐范式 | 拓扑范式 (TIAE) |
|---|---|
| 逐点学习意图→动作映射 | 学习距离保持嵌入 |
| 需要穷举覆盖意图空间 | 通过结构泛化到未见意图 |
| 依赖搜索的推理 (CEM/MPPI) | 无搜索直接控制 |
| O(N) 或 O(N²) 复杂度 | O(1) 推理 |
| 对分布偏移脆弱 | 对新意图和形态鲁棒 |

### 核心特性

- **拓扑保持损失**：最小化意图空间和动作空间中归一化成对距离矩阵的平方误差
- **双通道编码**：状态和意图特征的分离处理与交叉注意力融合
- **非对称梯度路由**：local intent 附加梯度；goal intent 使用 stop-gradient
- **无搜索直接控制**：2.9–5.5 ms 延迟，零候选搜索
- **零样本泛化**：相比对齐基线，未见目标泛化能力提升 2.3 倍

---

## 架构

### TIAE 框架

```
                    ┌────────────────────────────┐
 示范轨迹 ───►       │     视觉编码器 (ViT)         │
                    │   RGB → 潜在变量 z ∈ ℝᵈ    │
                    └──────┬──────────┬──────────┘
                           │          │
              ┌────────────▼──┐  ┌─────▼──────────────┐
              │ 前向 JEPA 预测器│  │  意图-动作 Actor    │
              │ z_t → z_{t+1} │  │  双通道编码          │
              └────────┬──────┘  │  m_t → p(a_t|z_t,m_t)│
                       │         └─────┬────────────────┘
                       │               │
        ┌──────────────▼───────────────▼────────────────┐
        │              拓扑保持损失                        │
        │  L_topo = Σ (d_intent(i,j) - d_action(μ_i,μ_j))²│
        │  双重实例:  m_local = z_{t+1} - z_t             │
        │            m_goal  = sg(z_g) - z_t              │
        └────────────────────────────────────────────────┘
```

### 模块架构

#### 表 1：视觉编码器

| 层 | 配置 | 输出维度 |
|----|------|---------|
| ViT 骨干 | ViT-S/16, 224×224 输入 | 384 |
| 投影器 | Linear → LayerNorm → GELU → Linear | 192 |

#### 表 2：前向 JEPA 预测器

| 组件 | 配置 |
|------|------|
| 架构 | 条件 Transformer |
| 深度 | 6 个模块 |
| 注意力头 | 8 |
| 模型维度 | 384 |
| 前馈维度 | 1536 |
| 上下文窗口 | 8 帧 |
| 条件注入 | 通过调制（缩放+偏移+门控）注入动作嵌入 |

#### 表 3：意图-动作 Actor

| 组件 | 配置 |
|------|------|
| 特征语法 | [z_t, m_t, z_t ⊙ m_t, A(a_{t-1})] |
| 架构 | 带残差 GELU 块的 MLP |
| 隐藏维度 | 1024 |
| 层数 | 3 |
| 输出 | 高斯分布 (μ, log σ), 2 × action_dim |
| Log std 范围 | [-5.0, 2.0] |

#### 表 4：拓扑保持模块

| 组件 | 配置 |
|------|------|
| 距离度量 | Cosine (默认) / Euclidean / Mahalanobis |
| 子集大小 | 256 (计算效率优化) |
| 归一化 | 尺度不变的 [0, 1] 归一化 |
| 损失权重 | 0.05 (displacement 变体) |

#### 表 5：双通道编码器（可选）

| 组件 | 配置 |
|------|------|
| 状态投影 | Linear(hidden_dim) + LayerNorm + GELU |
| 意图投影 | Linear(hidden_dim) + LayerNorm + GELU |
| 交叉注意力 | 4-头 MHA, 意图查询状态 |
| 融合 | Concat + 2 层 MLP |

### 损失函数

完整的 TIAE 目标：

$$
\mathcal{L}_{\text{TIAE}} = \mathcal{L}_{\text{JEPA}} + \lambda_{\text{sig}} \mathcal{L}_{\text{SIGReg}} + \lambda_{\text{local}} \mathcal{L}_{\text{local}} + \lambda_{\text{goal}} \mathcal{L}_{\text{goal}} + \lambda_{\text{topo}} \mathcal{L}_{\text{topo}}
$$

其中拓扑保持损失：

$$
\mathcal{L}_{\text{topo}} = \frac{1}{N(N-1)} \sum_{i \neq j} \left( \frac{d_M(i,j)}{\max d_M} - \frac{d_A(\mu_i, \mu_j)}{\max d_A} \right)^2
$$

---

## 安装

### 环境要求

- Ubuntu 22.04
- Python 3.10
- PyTorch 2.6.0+
- CUDA 12.4
- NVIDIA GPU (≥ 24 GB 显存推荐)

### 环境配置

```bash
git clone https://github.com/molyswu/TIAE.git
cd TIAE
bash scripts/install.sh cu124
source .venv/bin/activate
cp .env.example .env
# 编辑 .env 设置 STABLEWM_HOME 和 LOCAL_DATASET_DIR
```

仅做导入和配置检查：
```bash
bash scripts/install.sh cpu
```

### 数据准备

数据来自 [LeWM Hugging Face 合集](https://huggingface.co/collections/quentinll/lewm)。
放置在 `$LOCAL_DATASET_DIR/datasets/` 下：

```text
$LOCAL_DATASET_DIR/
└── datasets/
    ├── pusht_expert_train.lance   # PushT (从 HDF5 转换)
    ├── ogbench/cube_single_expert.h5
    ├── reacher.h5
    └── tworoom.h5
```

PushT 转换为 Lance 格式：
```bash
python -m stable_worldmodel.cli convert \
  pusht_expert_train pusht_expert_train.lance \
  --source-format hdf5 --dest-format lance
```

---

## 训练

### 单任务训练

每种任务训练一个 TIAE 模型（1 epoch, 3 seeds）：

```bash
# PushT — displacement intent（默认）
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config-name=tiae_goal \
  data=pusht \
  output_model_name=tiae_goal_pusht_s3072 \
  seed=3072

# Cube
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config-name=tiae_goal \
  data=ogb \
  output_model_name=tiae_goal_cube_s3072 \
  seed=3072

# Reacher
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config-name=tiae_goal \
  data=dmc \
  output_model_name=tiae_goal_reacher_s3072 \
  seed=3072

# TwoRoom
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config-name=tiae_goal \
  data=tworoom \
  output_model_name=tiae_goal_tworoom_s3072 \
  seed=3072
```

训练过程日志输出：

```text
TRAIN_PROGRESS={"epoch": 1, "global_step": 100, "loss": ..., "topo_loss": ..., "lr": ...}
```

### 多任务共享编码器训练

每任务一张 GPU，共享一个视觉编码器：

```bash
torchrun --nproc_per_node=4 train_multitask.py \
  --variant displacement \
  --seed 3072 \
  --epochs 5 \
  --output-cache $STABLEWM_HOME/checkpoints \
  --run-dir outputs/tiae_multitask_goal_s3072_e5
```

### 拓扑损失权重调节

| 变体 | topo_weight | 效果 |
|------|-------------|------|
| `displacement` | 0.05 | 完整 TIAE（推荐） |
| `goal_only` | 0.03 | 仅目标监督 + 弱拓扑 |
| `inverse` | 0.03 | 仅局部监督 + 弱拓扑 |
| `lewm` | 0.0 | 基线 JEPA（无拓扑） |

---

## 评估

### 评估模式

| 模式 | 求解器 | 搜索 | 描述 |
|------|--------|------|------|
| `direct` | `DirectSolver` | 无 | 无搜索直接控制 |
| `pure_cem` | `PureCEMSolver` | CEM 300×30 | 纯表示基线 |
| `actor_cem` | `ActorCEMSolver` | CEM | 以 actor 为中心的可选验证 |
| `guarded_a` | `GuardedCEMSolver` | CEM 128×3 | Direct 为中心的带保护验证 |

### LeWM 协议评估

```bash
# 无搜索直接评估
bash scripts/eval_official.sh direct pusht \
  tiae_goal_pusht_s3072/weights_epoch_1.pt 42 100

# CEM 基线
bash scripts/eval_official.sh pure_cem pusht \
  tiae_goal_pusht_s3072/weights_epoch_1.pt 42 100

# 带保护验证
bash scripts/eval_official.sh guarded_a pusht \
  tiae_goal_pusht_s3072/weights_epoch_1.pt 42 100
```

### 零样本动力学泛化

```bash
# 插值：评估未见过的中间目标
python eval.py eval=pusht-zero-shot zero_shot_mode=interpolation \
  policy=${CHECKPOINT_PATH} seed=42

# 外推：评估训练分布外的目标
python eval.py eval=pusht-zero-shot zero_shot_mode=extrapolation \
  policy=${CHECKPOINT_PATH} seed=42

# 意图扰动：评估对噪声意图的鲁棒性
python eval.py eval=pusht-zero-shot zero_shot_mode=perturbation \
  policy=${CHECKPOINT_PATH} seed=42 perturbation_sigma=0.1

# 跨任务迁移
python eval.py eval=pusht-zero-shot zero_shot_mode=cross_task \
  policy=${CHECKPOINT_PATH} seed=42
```

---

## 结果

### 单任务，1 Epoch

| 任务 | Direct SR (%) | Guarded-A SR (%) | 推理延迟 (ms) |
|------|--------------|------------------|--------------|
| PushT | 96.0 ± 0.5 | 97.3 ± 0.4 | 2.9 |
| Cube | 95.7 ± 0.6 | 97.8 ± 0.3 | 3.5 |
| Reacher | 97.3 ± 0.4 | 98.5 ± 0.3 | 4.1 |
| TwoRoom | 92.3 ± 0.8 | 96.2 ± 0.5 | 5.5 |
| **Macro** | **95.3 ± 0.6** | **97.5 ± 0.4** | **4.0** |

### 消融实验

| 变体 | Direct SR (%) | Guarded-A SR (%) |
|------|--------------|------------------|
| TIAE (完整) | **97.2 ± 0.8** | 98.1 ± 0.7 |
| + TopoLoss only | 88.4 ± 1.5 | 90.1 ± 1.3 |
| + DualEnc only | 85.2 ± 1.8 | 87.6 ± 1.6 |
| 对齐基线 | 67.0 ± 2.1 | 68.3 ± 2.0 |

### 零样本泛化

| 条件 | TIAE (%) | 对齐基线 (%) | 提升 (pp) |
|------|---------|------------|----------|
| 分布内插值 | 94.5 ± 1.1 | 62.7 ± 2.3 | +31.8 |
| 外推 | 88.3 ± 1.5 | 41.2 ± 3.1 | +47.1 |
| 意图扰动 (σ=0.1) | 92.1 ± 1.3 | 58.4 ± 2.7 | +33.7 |
| 跨任务迁移 | 89.6 ± 1.4 | 61.3 ± 2.9 | +28.3 |

---

## 论文

**"From Intent Alignment to Intent Topology: A Topological Theory of Goal-Directed Embodied Control"**

论文 PDF 包含在本仓库中：`Manuscript_Wu_et_al_Topological_Theory.pdf`

---

## 引用

```bibtex
@article{wu2026tiae,
  title   = {From Intent Alignment to Intent Topology:
             A Topological Theory of Goal-Directed Embodied Control},
  author  = {Wu, Yuxiang and Wu, Tusun and Wu, Yuyan},
  year    = {2026},
  journal = {Preprint}
}
```

---

问题咨询：<luoliibaqi4747@gmail.com> · [欢迎提交 Issue](https://github.com/molyswu/TIAE/issues)
