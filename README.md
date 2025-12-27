# ECSPNet Implementation

**Paper-exact implementation of "Deep Reinforcement Learning Energy Scheduling"**

## VERSION: 2.0-GPU - Full GPU Acceleration

This version runs the entire training loop on GPU:
- Environment simulation (GPUBatchECSPEnv) runs entirely on GPU using PyTorch tensors
- No CPU-GPU data transfers during rollouts
- Baseline computation on GPU
- Expected 10-50x speedup over CPU-bound version

## Overview

This implementation reproduces the ECSPNet method for solving the Energy-Conscious Scheduling Problem (ECSP) using deep reinforcement learning with a transformer-based policy network.

## Paper Parameters (Fixed)

| Parameter | Value |
|-----------|-------|
| Benchmark scales | N = [20, 40, 60, 100] |
| TOU pattern | [High:0.6, Low:0.4, High:0.6, Low:0.4] time units |
| Time unit | 10 hours |
| Step durations | Step1∈[0.4,0.8], Step2∈[0.2,0.6], Step3∈[0.4,0.8] |
| P_high | 1.0 (fixed) |
| Max wait (T_PW) | 0.4 time units |
| Training epochs | 3000 |
| Batch size | 2048 |
| Learning rate | 1e-3 (÷10 at epochs 1000, 2000) |
| Dataset | 50×2048 new instances per epoch |
| Inference B | 1000 solutions |
| Truncation β | 0.9 |
| Entropy α | 0.1 |
| d_model | 128 |
| ECSP blocks | 2 |

## Installation

```bash
pip install torch numpy gymnasium tqdm matplotlib
```

## Project Structure

```
ecsp/
├── __init__.py     # Package exports
├── data.py         # Instance generation & TOU patterns
├── env.py          # ECSPEnv Gym environment
├── model.py        # ECSPNet architecture
├── train.py        # Algorithm 1 - Training loop
├── infer.py        # Algorithm 2 - Inference & Pareto
└── main.py         # CLI entry point
```

## Usage

### Training

Train on all benchmark scales:
```bash
python -m ecsp.main train --epochs 3000 --device cuda
```

Train on specific scales:
```bash
python -m ecsp.main train --scales 20 40 --epochs 1000
```

### Evaluation

Evaluate trained models:
```bash
python -m ecsp.main eval --checkpoint-dir checkpoints
```

### Demo

Run demo on a single instance:
```bash
python -m ecsp.main demo --N 20 --visualize
```

### Python API

```python
from ecsp import ECSPEnv, ECSPNet, Trainer, Inferencer, generate_instance

# Generate instance
tasks = generate_instance(N=20)

# Create environment
env = ECSPEnv(N=20)
obs, info = env.reset(options={'tasks': tasks, 'w': 0.5})

# Create model
model = ECSPNet(d_model=128, num_heads=8, num_blocks=2)

# Train
trainer = Trainer(N=20, device='cuda')
trainer.train()

# Inference
inferencer = Inferencer(model, B=1000, beta=0.9)
pareto_front, _ = inferencer.solve_instance(tasks)
```

## Key Components

### Environment (ECSPEnv)

- **State**: tasks[N,5], EP[20], objs[2], w[1], mask[N]
- **Action**: Categorical(2N) → (task_idx, mode)
- **Mode 0**: No wait after step1
- **Mode 1**: Wait after step1 (capped at T_PW=0.4)

### Model (ECSPNet)

```
Input → Preprocessing → [ECSP Block ×2] → Output Head

Preprocessing:
1. tasks_emb = Linear(tasks) → [B, N, 128]
2. EP_emb = Linear(EP) → [B, 20, 128]
3. objs_emb = Linear(objs) → [B, 128]
4. w_emb = Linear(w) → [B, 128]
5. tasks_emb += objs_emb + w_emb (broadcast)

ECSP Block:
1. Cross-attention: Q=tasks, K/V=EP
2. Self-attention: Q/K/V=tasks

Output: softmax(Linear(tasks).flatten()) → [B, 2N]
```

### Training (Algorithm 1)

- REINFORCE with baseline
- Reward: R = -max(w×TWT, (1-w)×EEC)
- Baseline: 10 w-bins, b = T_pt × mean(R/T_pt in bin)
- Entropy regularization: α = 0.1

### Inference (Algorithm 2)

- B=1000 solutions with w = i/(B+1)
- Truncation sampling: P' = ReLU(P - β×max(P))
- Pareto front filtering

## Validation Checklist

- [x] State: tasks[5D], EP[20], objs[2], w[1], mask[N]
- [x] Action: Categorical(2N) → (task_idx, mode)
- [x] Wait: cap at 0.4, force step2 if still high-price
- [x] EEC: slot-overlap PH[k]×dt during step2, ×2 at end
- [x] Network: embed→add objs/w→2 blocks→(N,2)→flatten softmax
- [x] Reward: R = -max(w×TWT, (1-w)×EEC)
- [x] Baseline: 10 w-bins, b = T_pt × mean(R/T_pt in bin)
- [x] Training: 2048 batch, 3000 epochs, exact LR schedule
- [x] Inference: B=1000, β=0.9, w=i/1001, ReLU truncation
