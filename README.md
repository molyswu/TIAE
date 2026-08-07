<div align="center">
  <h1>TIAE: Topological Intent-Action Embedding</h1>
  <h3>A Topological Theory of Goal-Directed Embodied Control</h3>

  <p><strong>Learn a topology-preserving map from the intent manifold to the action manifold — no alignment, no search, just structure.</strong></p>

  <p>
    Yuxiang Wu<sup>1*</sup> &nbsp;&middot;&nbsp;
    Tusun Wu<sup>2*</sup> &nbsp;&middot;&nbsp;
    Yuyan Wu<sup>3</sup>
  </p>
  <p>
    <sup>1</sup>The Education University of Hong Kong<br>
    <sup>2</sup>Shenzhen Metachip Technology Co., Ltd.<br>
    <sup>3</sup>Guangdong Polytechnic Normal University<br>
    <sup>*</sup>Corresponding authors
  </p>

  <p>
    <a href="#overview">Overview</a> &nbsp;&middot;&nbsp;
    <a href="#architecture">Architecture</a> &nbsp;&middot;&nbsp;
    <a href="#installation">Installation</a> &nbsp;&middot;&nbsp;
    <a href="#training">Training</a> &nbsp;&middot;&nbsp;
    <a href="#evaluation">Evaluation</a> &nbsp;&middot;&nbsp;
    <a href="#results">Results</a>
  </p>
</div>

---

## Overview

**TIAE (Topological Intent-Action Embedding)** is a JEPA-based world model
framework that learns goal-directed control by **preserving the topology** of
the intent manifold in the action manifold. Instead of aligning specific intents
to specific actions (the "alignment assumption"), TIAE preserves pairwise
distance structure — similar intents induce similar action distributions,
distant intents map to distinct action laws.

### Why Topology > Alignment?

| Alignment Paradigm | Topology Paradigm (TIAE) |
|---|---|
| Learn point-wise intent→action mapping | Learn distance-preserving embedding |
| Requires exhaustive coverage of intent space | Generalizes to unseen intents via structure |
| Search-dependent inference (CEM/MPPI) | Search-free direct control |
| O(N) or O(N²) complexity | O(1) inference |
| Brittle to distribution shift | Robust to novel intents and embodiments |

### Key Features

- **Topology-Preserving Loss**: Minimizes squared error between normalized
  pairwise distance matrices in intent space and action space
- **Dual-Channel Encoding**: Separate processing of state and intent features
  with cross-attention fusion
- **Asymmetric Gradient Routing**: Local intent attaches gradients; goal intent
  uses stop-gradient for deployment anchoring
- **Search-Free Direct Control**: 2.9–5.5 ms latency with zero candidate search
- **Zero-Shot Generalization**: 2.3× better generalization to unseen goals vs
  alignment baselines

---

## Architecture

### TIAE Framework

```
                    ┌─────────────────────────────────┐
 Demonstration       │           Vision Encoder (ViT)    │
Trajectories  ───►  │        RGB → latent z ∈ ℝᵈ       │
                    └───────┬─────────────┬─────────────┘
                            │             │
               ┌────────────▼──┐   ┌──────▼──────────────┐
               │ Forward JEPA   │   │  Intent-Action Actor │
               │  Predictor     │   │  Dual-Channel Encode  │
               │ z_t → z_{t+1} │   │  m_t → p(a_t|z_t,m_t) │
               └────────┬───────┘   └──────┬───────────────┘
                        │                  │
         ┌──────────────▼──────────────────▼────────────────┐
         │              Topology-Preserving Loss              │
         │  L_topo = Σ (d_intent(i,j) - d_action(μ_i,μ_j))²  │
         │  Dual instances:  m_local = z_{t+1} - z_t         │
         │                   m_goal  = sg(z_g) - z_t         │
         └───────────────────────────────────────────────────┘
```

### Component Architecture

#### Table 1: Vision Encoder

| Layer | Configuration | Output Dim |
|-------|--------------|------------|
| ViT Backbone | ViT-S/16, 224×224 input | 384 |
| Projector | Linear → LayerNorm → GELU → Linear | 192 |

#### Table 2: Forward JEPA Predictor

| Component | Configuration |
|-----------|--------------|
| Architecture | Conditional Transformer |
| Depth | 6 blocks |
| Heads | 8 |
| Model dim | 384 |
| Feedforward dim | 1536 |
| Context window | 8 frames |
| Conditioning | Action embedding via modulation (scale + shift + gate) |

#### Table 3: Intent-Action Actor

| Component | Configuration |
|-----------|--------------|
| Feature Grammar | [z_t, m_t, z_t ⊙ m_t, A(a_{t-1})] |
| Architecture | MLP with residual GELU blocks |
| Hidden dim | 1024 |
| Depth | 3 layers |
| Output | Gaussian (μ, log σ), 2 × action_dim |
| Log std range | [-5.0, 2.0] |

#### Table 4: Topology-Preserving Module

| Component | Configuration |
|-----------|--------------|
| Distance metric | Cosine (default) / Euclidean / Mahalanobis |
| Subset size | 256 (for computational efficiency) |
| Normalization | Scale-invariant [0, 1] normalization |
| Loss weight | 0.05 (displacement variant) |

#### Table 5: Dual-Channel Encoder (optional)

| Component | Configuration |
|-----------|--------------|
| State projection | Linear(hidden_dim) + LayerNorm + GELU |
| Intent projection | Linear(hidden_dim) + LayerNorm + GELU |
| Cross-attention | 4-head MHA, intent queries state |
| Fusion | Concat + 2-layer MLP |

### Loss Function

The full TIAE objective:

$$
\mathcal{L}_{\text{TIAE}} = \mathcal{L}_{\text{JEPA}} + \lambda_{\text{sig}} \mathcal{L}_{\text{SIGReg}} + \lambda_{\text{local}} \mathcal{L}_{\text{local}} + \lambda_{\text{goal}} \mathcal{L}_{\text{goal}} + \lambda_{\text{topo}} \mathcal{L}_{\text{topo}}
$$

Where:

$$
\mathcal{L}_{\text{local}} = -\log p_\eta(a_t \mid z_t, z_{t+1}-z_t, A(a_{t-1}))
$$

$$
\mathcal{L}_{\text{goal}} = -\log p_\eta(a_t \mid z_t, \text{sg}(z_g)-z_t, A(a_{t-1}))
$$

$$
\mathcal{L}_{\text{topo}} = \frac{1}{N(N-1)} \sum_{i \neq j} \left( \frac{d_M(i,j)}{\max d_M} - \frac{d_A(\mu_i, \mu_j)}{\max d_A} \right)^2
$$

---

## Installation

### Requirements

- Ubuntu 22.04
- Python 3.10
- PyTorch 2.6.0+
- CUDA 12.4
- NVIDIA GPU (≥ 24 GB VRAM recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/molyswu/TIAE.git
cd TIAE

# Install with CUDA 12.4 support
bash scripts/install.sh cu124
source .venv/bin/activate

# Configure environment
cp .env.example .env
# Edit .env: set STABLEWM_HOME and LOCAL_DATASET_DIR
```

For CPU-only import/configuration checks:
```bash
bash scripts/install.sh cpu
```

### Data

TIAE uses datasets from the [LeWM Hugging Face collection](https://huggingface.co/collections/quentinll/lewm).
Place them under `$LOCAL_DATASET_DIR/datasets/`:

```text
$LOCAL_DATASET_DIR/
└── datasets/
    ├── pusht_expert_train.lance   # PushT (convert from HDF5)
    ├── ogbench/cube_single_expert.h5  # Cube
    ├── reacher.h5                     # Reacher
    └── tworoom.h5                     # TwoRoom
```

Convert PushT to Lance format:
```bash
python -m stable_worldmodel.cli convert \
  pusht_expert_train pusht_expert_train.lance \
  --source-format hdf5 --dest-format lance
```

---

## Training

### Single-Task Training

Train a task-specific TIAE model (1 epoch, 3 seeds for paper results):

```bash
# PushT — displacement intent (default)
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config-name=tiae_goal \
  data=pusht \
  output_model_name=tiae_goal_pusht_s3072 \
  seed=3072

# Cube — displacement intent
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config-name=tiae_goal \
  data=ogb \
  output_model_name=tiae_goal_cube_s3072 \
  seed=3072

# Reacher — displacement intent
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config-name=tiae_goal \
  data=dmc \
  output_model_name=tiae_goal_reacher_s3072 \
  seed=3072

# TwoRoom — displacement intent
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config-name=tiae_goal \
  data=tworoom \
  output_model_name=tiae_goal_tworoom_s3072 \
  seed=3072

# Waypoint variant (speed-aware intents)
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config-name=tiae_waypoint \
  data=pusht \
  seed=3072
```

Training progress is logged at every step:

```text
TRAIN_PROGRESS={"epoch": 1, "global_step": 100, "loss": ..., "topo_loss": ..., "lr": ...}
```

### Multi-Task Training (Shared Encoder)

Train with one shared encoder across 4 tasks, each on one GPU:

```bash
torchrun --nproc_per_node=4 train_multitask.py \
  --variant displacement \
  --seed 3072 \
  --epochs 5 \
  --output-cache $STABLEWM_HOME/checkpoints \
  --run-dir outputs/tiae_multitask_goal_s3072_e5
```

For the LeWM baseline (no intent-action head or topology loss):

```bash
torchrun --nproc_per_node=4 train_multitask.py \
  --variant lewm \
  --seed 3072 \
  --epochs 5 \
  --output-cache $STABLEWM_HOME/checkpoints \
  --run-dir outputs/tiae_multitask_lewm_s3072_e5
```

### Topology Loss Weight Tuning

The topology-preserving loss weight (`topo_weight`) can be adjusted to control
the strength of manifold structure preservation:

| Variant | topo_weight | Effect |
|---------|-------------|--------|
| `displacement` | 0.05 | Full TIAE (recommended) |
| `goal_only` | 0.03 | Goal-only supervision + weak topology |
| `inverse` | 0.03 | Local-only supervision + weak topology |
| `lewm` | 0.0 | Baseline JEPA (no topology) |

---

## Evaluation

### Evaluation Modes

| Mode | Solver | Search | Description |
|------|--------|--------|-------------|
| `direct` | `DirectSolver` | None | **Search-free** topology-preserving control |
| `pure_cem` | `PureCEMSolver` | CEM 300×30 | Actor-disabled representation baseline |
| `actor_cem` | `ActorCEMSolver` | CEM | Actor-centered optional verification |
| `guarded_a` | `GuardedCEMSolver` | CEM 128×3 | Direct-centered local verification |

### Official LeWM Protocol

```bash
# Direct (search-free) evaluation
bash scripts/eval_official.sh direct pusht \
  tiae_goal_pusht_s3072/weights_epoch_1.pt 42 100

# Pure CEM baseline
bash scripts/eval_official.sh pure_cem pusht \
  tiae_goal_pusht_s3072/weights_epoch_1.pt 42 100

# Guarded A verification
bash scripts/eval_official.sh guarded_a pusht \
  tiae_goal_pusht_s3072/weights_epoch_1.pt 42 100
```

### CLEAR-LeWM v0.5.1 Protocol

```bash
export CLEAR_LEWM_ROOT=/path/to/CLEAR-LeWM-v0.5.1
export CLEAR_LEWM_PYTHON="$CLEAR_LEWM_ROOT/.venv/bin/python"
CHECKPOINT=tiae_goal_pusht_s3072/weights_epoch_1.pt

bash scripts/eval_clear_v051.sh direct pusht \
  "$CHECKPOINT" "$STABLEWM_HOME/datasets/pusht_expert_train.h5" \
  "$CLEAR_LEWM_ROOT" /path/to/LeWM 42 \
  results/pusht-tiae-direct.json
```

### Zero-Shot Dynamics Generalization

```bash
# Interpolation: evaluate on unseen intermediate goals
python eval.py eval=pusht-zero-shot-zero_shot_mode=interpolation \
  policy=${CHECKPOINT_PATH} seed=42

# Extrapolation: evaluate on goals outside training distribution
python eval.py eval=pusht-zero-shot-zero_shot_mode=extrapolation \
  policy=${CHECKPOINT_PATH} seed=42

# Intent perturbation: evaluate robustness to noisy intents
python eval.py eval=pusht-zero-shot-zero_shot_mode=perturbation \
  policy=${CHECKPOINT_PATH} seed=42 perturbation_sigma=0.1

# Cross-task transfer: leave-one-out evaluation
python eval.py eval=pusht-zero-shot-zero_shot_mode=cross_task \
  policy=${CHECKPOINT_PATH} seed=42
```

---

## Results

### Task-Specific, 1 Epoch (Displacement Intent)

| Task | Direct SR (%) | Guarded-A SR (%) | CEM 300×30 SR (%) | Latency (ms) |
|------|--------------|------------------|--------------------|--------------|
| PushT | 96.0 ± 0.5 | 97.3 ± 0.4 | 91.2 ± 1.2 | 2.9 |
| Cube | 95.7 ± 0.6 | 97.8 ± 0.3 | 90.5 ± 1.5 | 3.5 |
| Reacher | 97.3 ± 0.4 | 98.5 ± 0.3 | 93.1 ± 1.0 | 4.1 |
| TwoRoom | 92.3 ± 0.8 | 96.2 ± 0.5 | 88.7 ± 1.6 | 5.5 |
| **Macro** | **95.3 ± 0.6** | **97.5 ± 0.4** | **90.9 ± 1.3** | **4.0** |

### Shared Encoder, 5 Epochs (Displacement Intent)

| Variant | Direct SR (%) | CEM SR (%) |
|---------|--------------|------------|
| TIAE (Full) | **97.2 ± 0.8** | 98.1 ± 0.7 |
| + TopoLoss only | 88.4 ± 1.5 | 90.1 ± 1.3 |
| + DualEnc only | 85.2 ± 1.8 | 87.6 ± 1.6 |
| Alignment Baseline | 67.0 ± 2.1 | 68.3 ± 2.0 |

### Zero-Shot Generalization

| Condition | Alignment Baseline (%) | TIAE (%) | Δ (pp) |
|-----------|----------------------|----------|--------|
| Same Distribution | 85.3 ± 1.2 | 97.2 ± 0.8 | +11.9 |
| Interpolation (unseen gaps) | 62.7 ± 2.3 | 94.5 ± 1.1 | +31.8 |
| Extrapolation (outside dist.) | 41.2 ± 3.1 | 88.3 ± 1.5 | +47.1 |
| Intent Perturbation (σ=0.1) | 58.4 ± 2.7 | 92.1 ± 1.3 | +33.7 |
| Intent Perturbation (σ=0.3) | 32.8 ± 3.5 | 85.6 ± 1.8 | +52.8 |
| Cross-Task Transfer | 61.3 ± 2.9 | 89.6 ± 1.4 | +28.3 |
| Novel Embodiment | 52.7 ± 3.2 | 86.4 ± 1.6 | +33.7 |

### Theory-Measurement Alignment

Across 45 shared-encoder checkpoints, the topology-preserving relation
between intent and action-law families strongly predicts Direct SR:

| Metric | Correlation with Direct SR (r) |
|--------|-------------------------------|
| Predicted-expert kNN overlap | 0.954 |
| Action-family linear CKA | 0.897 |
| Topology loss (negated) | 0.883 |
| Pointwise action R² | 0.815 |

---

## Paper

**"From Intent Alignment to Intent Topology: A Topological Theory of Goal-Directed Embodied Control"**

Yuxiang Wu, Tusun Wu, Yuyan Wu

Published 2026.

**Paper PDF**: Included in this repository as `Manuscript_Wu_et_al_Topological_Theory.pdf`

---

## Citation

```bibtex
@article{wu2026tiae,
  title   = {From Intent Alignment to Intent Topology:
             A Topological Theory of Goal-Directed Embodied Control},
  author  = {Wu, Yuxiang and Wu, Tusun and Wu, Yuyan},
  year    = {2026},
  journal = {Preprint}
}
```

Please also reference the machine-readable record in [`CITATION.cff`](CITATION.cff).

---

## Project Structure

```text
TIAE/
├── tiae.py                  # TIAE model: topology-preserving intent-action embedding
├── jepa.py                  # Base JEPA world model
├── module.py                # Neural modules: TopologyLoss, DualChannel, SIGReg, etc.
├── train.py                 # Single-task training with topology loss
├── train_multitask.py       # Multi-task shared-encoder training
├── eval.py                  # Evaluation with zero-shot generalization tests
├── direct_solver.py         # Search-free direct control solver
├── cem_solvers.py           # CEM-based solvers (pure, actor-CEM)
├── abcd_solvers.py          # Guarded-A with local verification
├── prior_only_solver.py     # CLEAR-LeWM direct solver
├── utils.py                 # Dataset utilities, preprocessing, checkpointing
├── config/                  # Hydra configs for all variants and tasks
│   └── train/               # Training configs: tiae_goal, tiae_waypoint, etc.
├── scripts/                 # Shell launchers for training, evaluation, installation
├── tests/                   # Configuration composition tests
├── tools/                   # Build and export utilities
├── docs/                    # Documentation and methodology notes
└── paper_runtime/           # Isolated runtime for published checkpoints
```

---

## Community & Contributions

Issues and pull requests are welcome. For questions:

- Email: <wts20230305@gmail.com>
- [GitHub Issues](https://github.com/molyswu/TIAE/issues)

---

## License

See [LICENSE](LICENSE) for the code license and [NOTICE.md](NOTICE.md) for
provenance and third-party licensing boundaries.
