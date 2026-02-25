# Getting Started with Reinforcement Learning Strategies

**A complete, beginner-friendly guide to training and running the RL strategies in this 3D bin packing project.**

If you have never worked with RL or machine learning before, this guide is for you. Every step includes exact commands you can copy-paste, and every technical term is explained in plain English before it is used.

---

## Table of Contents

1. [What is Reinforcement Learning?](#1-what-is-reinforcement-learning)
2. [Our 5 RL Strategies — Plain English](#2-our-5-rl-strategies--plain-english)
3. [Prerequisites — What You Need Installed](#3-prerequisites--what-you-need-installed)
4. [Quick Start — Run Locally in 5 Minutes](#4-quick-start--run-locally-in-5-minutes)
5. [Training on HPC — Complete Step-by-Step](#5-training-on-hpc--complete-step-by-step)
6. [Understanding the Output](#6-understanding-the-output)
7. [Troubleshooting FAQ](#7-troubleshooting-faq)
8. [Which Strategy Should I Train First?](#8-which-strategy-should-i-train-first)
9. [File Structure Reference](#9-file-structure-reference)

---

## 1. What is Reinforcement Learning?

Imagine you are training a new warehouse worker. On their first day, they have no idea where to place boxes on a pallet. They try placing a box somewhere and sometimes it works well (the pallet stays flat, the box fits snugly) and sometimes it does not (the box hangs over the edge, or leaves a huge gap). Over hundreds of shifts, the worker starts to notice patterns: "When I put big boxes on the bottom, things work out better. When I fill corners first, I can fit more in." Nobody told them these rules explicitly. They learned by **trial and error**, getting feedback after each attempt.

That is exactly how Reinforcement Learning (RL) works. An "agent" (a computer program) makes decisions, receives a "reward" (a score that says how good that decision was), and gradually learns a "policy" (a strategy for making good decisions). The agent interacts with an "environment" (in our case, a simulation of the Botko BV packing station with a conveyor belt and two pallets) millions of times. Early on, the agent makes random, terrible decisions. Over time, it discovers that certain patterns lead to higher rewards and starts exploiting those patterns.

The key difference from traditional programming is that **we do not tell the agent what to do**. We only tell it what "good" looks like (high fill rate, stable placements, flat surfaces). The agent figures out the "how" entirely on its own. This is powerful because the agent can discover strategies that human engineers might never think of. The downside is that it takes a lot of computing time to train, because the agent needs millions of practice rounds.

---

## 2. Our 5 RL Strategies -- Plain English

We have implemented five different RL approaches. Each one tackles the same problem (packing boxes onto pallets) but uses a different learning method. Think of them as five different types of employees, each with a different way of thinking about the same job.

### 2.1 rl_dqn — "The Memory-Based Worker"

**Analogy**: Like a warehouse worker who keeps a notebook. Every time they try a placement and it works well, they write it down. Every time it fails, they note that too. When a new box arrives, they flip through their notebook and say: "Last time I had a box this size, putting it in that corner scored an 8 out of 10. Let me try that again."

**How it works**: The agent looks at the current state of both pallets (represented as "heightmaps" -- a top-down grid showing how high each spot on the pallet is) and the current box. It generates around 100-200 smart candidate positions (corners, edges, and key spots) rather than checking all 38,400 possible grid positions. For each candidate, it predicts a "Q-value" (how good it expects the final outcome to be if it places the box there). It picks the placement with the highest Q-value. Training uses a "replay buffer" -- a memory bank of past experiences that it re-studies over and over, like a student reviewing flashcards.

**Training time**: ~12 hours on a GPU.

### 2.2 rl_ppo — "The Chain of Decision-Makers"

**Analogy**: Like having four decision-makers in a chain at the loading dock. The first person looks at the situation and says "Put it on Pallet 1." The second person says "About 30cm from the left edge." The third says "About 20cm from the front." The fourth says "Turn it sideways." Each person only makes one small decision, but together they specify a complete placement.

**How it works**: Instead of choosing from thousands of possible placements all at once, this strategy breaks the decision into four smaller, easier decisions made one after another: (1) which pallet, (2) which X position, (3) which Y position, (4) which orientation. This reduces the action space from 38,400 choices down to just 204 choices (2 + 120 + 80 + 2). Each sub-decision is made by a separate "head" (output layer) of the neural network, and each head conditions on what the previous head decided. The "PPO" algorithm keeps training stable by preventing the agent from changing its behaviour too drastically in any single update.

**Training time**: ~16 hours on a GPU.

### 2.3 rl_a2c_masked — "The Safety-Conscious Worker"

**Analogy**: Like a worker who, before deciding where to put a box, first checks all the spots on the pallet and marks the ones that are physically impossible with red tape. "Can't go there -- it would hang over the edge. Can't go there -- there's nothing to support it. Can't go there -- another box is already there." Then they only consider the remaining valid spots when making their decision. This means they never waste time thinking about impossible placements.

**How it works**: The agent has a special "mask predictor" neural network that looks at the current state and instantly predicts which of the 1,536 possible actions are physically valid. Invalid actions get their probabilities set to zero so the agent never picks them. This mask is learned (not computed from scratch each time), making it very fast at inference time. The agent also uses "curriculum learning" -- it starts with easy scenarios (30 large boxes) and gradually progresses to hard ones (100 boxes of all sizes), like a student progressing through difficulty levels.

**Training time**: ~16 hours on a GPU.

### 2.4 rl_hybrid_hh — "The Manager Who Picks the Right Expert" (NOVEL -- thesis contribution)

**Analogy**: Like a shift manager at a warehouse who has seven expert workers, each with a different specialty. One is great at building flat layers. Another is great at filling corners. Another maximises surface contact. Instead of doing the work themselves, the manager looks at the current situation and decides: "For this box with this pallet state, I'll call Expert 3." The manager learns over time which expert handles which situation best.

**How it works**: This is the novel contribution of the thesis. Instead of learning WHERE to place boxes (which requires understanding complex 3D geometry), the agent learns WHICH existing heuristic strategy to call. It observes 39 handcrafted features about the current state (fill rates, roughness, box size, packing phase, etc.) and picks from 7 proven heuristics (baseline, walle_scoring, surface_contact, extreme_points, skyline, layer_building, best_fit_decreasing) plus a "skip" option. The selected heuristic then handles the actual placement. Because the action space is only 8 choices (instead of thousands), training is extremely fast and works on a regular CPU -- no GPU needed.

**Training time**: ~1-4 hours on CPU (no GPU required).

### 2.5 rl_pct_transformer — "The Spotlight Scanner"

**Analogy**: Like a worker who generates a list of all possible good spots (30-200 candidates), then shines a spotlight on each one in turn, examining them all simultaneously and in context of each other. Using a kind of "attention" mechanism, the worker considers how each candidate relates to every other candidate (e.g., "if I pick Spot A, it blocks Spot B later") and then points directly at the best one.

**How it works**: This strategy uses a "Transformer" neural network -- the same type of architecture that powers modern language models. First, a "candidate generator" produces 30-200 valid placement positions using corner points, extreme points, and floor scanning. Each candidate is described by 12 features (position, support, contact ratio, etc.). These candidates, plus information about the current box, are fed into a Transformer encoder that allows each candidate to "attend" to every other candidate. A "pointer decoder" then selects the single best candidate. This naturally handles the fact that different states have different numbers of valid placements.

**Training time**: ~16 hours on a GPU.

---

## 3. Prerequisites -- What You Need Installed

Before you can train any RL strategy, you need to have certain software installed on your computer. Here is what you need and how to check if you already have it.

### 3.1 Python (version 3.10 or higher)

**What it is**: Python is the programming language that all of our code is written in.

**Check if you have it**: Open a terminal (on Windows: search for "Git Bash" or "Command Prompt"; on Mac: open "Terminal") and type:

```bash
python --version
```

**Expected output** (your version might differ slightly, but it must be 3.10 or higher):
```
Python 3.10.12
```

If you do not have Python, download it from https://www.python.org/downloads/ and install it. During installation, make sure to check the box that says "Add Python to PATH".

### 3.2 pip (Python package installer)

**What it is**: pip is the tool that installs Python libraries (pre-made code written by other people that our project depends on).

**Check if you have it** (it comes with Python, but let us verify):

```bash
pip --version
```

**Expected output**:
```
pip 23.2.1 from /usr/lib/python3.10/site-packages/pip (python 3.10)
```

### 3.3 PyTorch (the machine learning framework)

**What it is**: PyTorch is the library that handles all the neural network math (matrix multiplications, gradient computations, GPU acceleration). Think of it as the engine that powers the learning.

**Install it** (if you do not have it yet):

If you have an NVIDIA GPU (for faster training):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

If you do NOT have an NVIDIA GPU (CPU-only, slower but works fine for rl_hybrid_hh):
```bash
pip install torch torchvision
```

**Verify it is installed**:
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

**Expected output** (GPU version):
```
PyTorch 2.2.0, CUDA available: True
```

**Expected output** (CPU-only version):
```
PyTorch 2.2.0, CUDA available: False
```

Both are fine. CUDA=True just means training will be faster for the GPU-heavy strategies.

### 3.4 All other dependencies

**What they are**: Our code uses several other libraries for logging, plotting, and environment simulation. The full list is in `strategies/rl_common/hpc/requirements.txt`.

**Install them all at once**:

```bash
cd "python/full workflow"
pip install -r strategies/rl_common/hpc/requirements.txt
```

**Expected output** (last few lines):
```
Successfully installed gymnasium-0.29.1 matplotlib-3.8.2 pandas-2.1.4 seaborn-0.13.0 tensorboard-2.15.1 tqdm-4.66.1 pyyaml-6.0.1 scipy-1.11.4
```

If some packages are already installed, pip will say "Requirement already satisfied" and that is perfectly fine.

### 3.5 Quick sanity check

Run this command to verify everything is correctly installed:

```bash
cd "python/full workflow"
python -c "
import torch
import gymnasium
import numpy as np
import matplotlib
import pandas
import tqdm
print('All imports successful!')
print(f'  PyTorch:    {torch.__version__}')
print(f'  CUDA:       {torch.cuda.is_available()}')
print(f'  Gymnasium:  {gymnasium.__version__}')
print(f'  NumPy:      {np.__version__}')
print(f'  Matplotlib: {matplotlib.__version__}')
print(f'  Pandas:     {pandas.__version__}')
"
```

**Expected output**:
```
All imports successful!
  PyTorch:    2.2.0
  CUDA:       True
  Gymnasium:  0.29.1
  NumPy:      1.26.3
  Matplotlib: 3.8.2
  Pandas:     2.1.4
```

If any import fails, go back and install the missing package with `pip install <package_name>`.

---

## 4. Quick Start -- Run Locally in 5 Minutes

This section gets you from zero to a trained model as fast as possible. We will use `rl_hybrid_hh` because it is the fastest strategy to train (minutes, not hours) and does not need a GPU.

### Step 1: Navigate to the project root

All commands in this guide assume you are in the "full workflow" directory:

```bash
cd "python/full workflow"
```

On Windows with the full path:
```bash
cd "C:/Users/Louis/Downloads/stapelalgortime/python/full workflow"
```

### Step 2: Verify the project imports work

This confirms that the entire packing framework loads correctly:

```bash
python -c "
import sys; sys.path.insert(0, '.')
from config import Box, BinConfig
from simulator.session import PackingSession, SessionConfig
from strategies.base_strategy import STRATEGY_REGISTRY
print(f'Framework loaded successfully!')
print(f'Available strategies: {len(STRATEGY_REGISTRY)}')
print(f'Strategy names: {sorted(STRATEGY_REGISTRY.keys())}')
"
```

**Expected output**:
```
Framework loaded successfully!
Available strategies: 24
Strategy names: ['baseline', 'best_fit_decreasing', ..., 'rl_dqn', 'rl_ppo', ...]
```

If you see `ModuleNotFoundError`, make sure you are in the right directory. See the [Troubleshooting FAQ](#7-troubleshooting-faq).

### Step 3: Run a quick 5-episode training of rl_hybrid_hh

This is a very short training run just to verify everything works. A real training run uses 10,000-50,000 episodes, but 5 episodes takes under a minute:

```bash
python strategies/rl_hybrid_hh/train.py --mode tabular --episodes 5 --output_dir outputs/rl_hybrid_hh_test
```

**What this command does**:
- `python strategies/rl_hybrid_hh/train.py` -- runs the training script
- `--mode tabular` -- uses the simple Q-table (fastest, no neural network)
- `--episodes 5` -- only trains for 5 episodes (a real run would be 10,000+)
- `--output_dir outputs/rl_hybrid_hh_test` -- saves results in this folder

**Expected output** (approximately):
```
[rl_hybrid_hh] Training config:
  Mode:     tabular
  Episodes: 5
  Epsilon:  1.0 -> 0.05
  ...

Episode 1/5 | Reward: 3.21 | Fill: 0.42 | Eps: 1.00 | Heuristics: baseline(3), walle(2)
Episode 2/5 | Reward: 4.15 | Fill: 0.48 | Eps: 0.81 | Heuristics: surface(4), walle(1)
Episode 3/5 | Reward: 3.87 | Fill: 0.45 | Eps: 0.61 | Heuristics: walle(3), baseline(2)
Episode 4/5 | Reward: 5.02 | Fill: 0.52 | Eps: 0.41 | Heuristics: walle(4), surface(1)
Episode 5/5 | Reward: 4.78 | Fill: 0.50 | Eps: 0.21 | Heuristics: walle(3), surface(2)

Training complete! Best fill: 0.52
Checkpoint saved: outputs/rl_hybrid_hh_test/checkpoints/best_model.pt
```

The exact numbers will differ because training involves randomness, but you should see fill rates between 0.3 and 0.6 and no errors.

### Step 4: Run a longer training (optional but recommended)

For a real training run that produces a useful model:

```bash
python strategies/rl_hybrid_hh/train.py --mode dqn --episodes 10000 --output_dir outputs/rl_hybrid_hh_run1
```

This takes approximately 1-4 hours on a modern CPU. You will see progress updates printed every 100 episodes. You can safely stop it at any time with Ctrl+C -- the best checkpoint is saved periodically.

### Step 5: Evaluate the trained model

After training completes, run the evaluation script:

```bash
python strategies/rl_hybrid_hh/evaluate.py --checkpoint outputs/rl_hybrid_hh_run1/checkpoints/best_model.pt --episodes 50
```

**Expected output**:
```
Evaluation Results (50 episodes):
  Mean fill rate:  0.621
  Std fill rate:   0.058
  Min fill rate:   0.482
  Max fill rate:   0.731

Heuristic selection distribution:
  walle_scoring:      38.2%
  surface_contact:    24.7%
  baseline:           15.3%
  extreme_points:      8.1%
  layer_building:      6.2%
  best_fit_decreasing: 4.1%
  skyline:             2.3%
  SKIP:                1.1%
```

### Step 6: Use the trained model in benchmark_all.py

The trained strategy can now be used like any other strategy in the framework:

```bash
python benchmark_all.py
```

The RL strategies that have trained checkpoints will automatically load and participate in the benchmark alongside the heuristic strategies.

---

## 5. Training on HPC -- Complete Step-by-Step

"HPC" stands for High-Performance Computing -- a university or company computer cluster with powerful GPUs. Training the GPU-based strategies (rl_dqn, rl_ppo, rl_a2c_masked, rl_pct_transformer) is much faster on an HPC because they have dedicated GPUs with large amounts of memory.

If you only want to train rl_hybrid_hh, you do NOT need HPC. It runs fine on your laptop.

### Step 1: Get access to your university's HPC

Contact your university's IT department or check your student portal for HPC access. You will receive:
- A **username** (e.g., `lstudent`)
- A **server address** (e.g., `hpc.university.edu`)
- A **password** or SSH key

### Step 2: Transfer files to the HPC

Open a terminal on your local machine and run:

```bash
scp -r "python/full workflow/" lstudent@hpc.university.edu:/home/lstudent/bin_packing/
```

**What this command does**:
- `scp` -- "secure copy", transfers files over the network
- `-r` -- recursive, meaning it copies folders and everything inside them
- The rest specifies "from here" and "to there"

Replace `lstudent` with your HPC username and `hpc.university.edu` with your server address.

**Expected output**:
```
config.py                                    100%   12KB   1.2MB/s   00:00
simulator/session.py                         100%   45KB   4.5MB/s   00:00
strategies/rl_dqn/train.py                   100%   28KB   2.8MB/s   00:00
... (many more files)
```

### Step 3: Connect to the HPC

```bash
ssh lstudent@hpc.university.edu
```

You will be prompted for your password. After entering it, you will see the HPC command prompt.

### Step 4: Set up the Python environment (one-time only)

This creates a "virtual environment" (an isolated Python installation) with all our dependencies:

```bash
cd /home/lstudent/bin_packing
bash strategies/rl_common/hpc/setup_hpc.sh
```

**What this does**:
1. Loads the Python and CUDA modules available on the HPC
2. Creates a virtual environment at `~/venvs/rl_packing`
3. Installs PyTorch with GPU support
4. Installs all other dependencies from `requirements.txt`
5. Runs a verification check

**Expected output** (last section):
```
-- Verification --
  Python:    3.10.8
  PyTorch:   2.2.0
  CUDA:      True
  GPU:       NVIDIA A100-SXM4-40GB
  GPU Mem:   40.0 GB
  Gymnasium: 0.29.1
  NumPy:     1.26.3
  Matplotlib:3.8.2
  All OK!

==================================================
  Setup complete! Activate with:
    source ~/venvs/rl_packing/bin/activate

  Then run training with:
    bash strategies/rl_common/hpc/train_all.sh
==================================================
```

If you get `module: command not found`, your HPC might use a different module system. Ask your HPC documentation which Python module to load and edit the `setup_hpc.sh` script accordingly.

### Step 5: Submit training jobs

The HPC uses a "job scheduler" called SLURM to manage who gets to use which GPUs. You submit a "job" and SLURM runs it when a GPU becomes available.

**To train all 5 strategies at once** (they run in parallel on different GPUs):

```bash
cd /home/lstudent/bin_packing
sbatch strategies/rl_common/hpc/train_all.sh
```

**Expected output**:
```
==================================================
  RL Strategy Training Launcher
  Strategies dir: /home/lstudent/bin_packing/strategies
  Output dir:     /home/lstudent/bin_packing/outputs/rl_training/20260222_140000
==================================================

-- rl_dqn --
  Submitted: Job ID 12345

-- rl_ppo --
  Submitted: Job ID 12346

-- rl_a2c_masked --
  Submitted: Job ID 12347

-- rl_hybrid_hh --
  Submitted: Job ID 12348

-- rl_pct_transformer --
  Submitted: Job ID 12349

==================================================
  Submitted 5 SLURM jobs:
    - Job 12345
    - Job 12346
    - Job 12347
    - Job 12348
    - Job 12349

  Monitor: squeue -u $USER
  Logs:    outputs/rl_training/20260222_140000/logs/
  Cancel:  scancel 12345 12346 12347 12348 12349
==================================================
```

**To train just one strategy** (e.g., rl_dqn only):

```bash
source ~/venvs/rl_packing/bin/activate
export PYTHONPATH="/home/lstudent/bin_packing:$PYTHONPATH"
cd /home/lstudent/bin_packing

python strategies/rl_dqn/train.py --episodes 50000 --batch_size 256 --lr 0.001
```

**Resource requirements per strategy**:

| Strategy | GPUs | CPUs | Memory | Wall Time |
|----------|------|------|--------|-----------|
| rl_dqn | 1 | 8 | 32 GB | ~12 hours |
| rl_ppo | 1 | 16 | 48 GB | ~16 hours |
| rl_a2c_masked | 1 | 16 | 48 GB | ~16 hours |
| rl_hybrid_hh | 0 (CPU only) | 8 | 16 GB | ~4 hours |
| rl_pct_transformer | 1 | 16 | 48 GB | ~16 hours |

### Step 6: Monitor training progress

While training is running, you can check on it:

**Check which jobs are running or queued**:
```bash
squeue -u $USER
```

**Expected output**:
```
  JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
  12345       gpu rl_rl_dq lstudent  R    2:15:30      1 gpu-node-03
  12346       gpu rl_rl_pp lstudent  R    2:15:28      1 gpu-node-04
  12347       gpu rl_rl_a2 lstudent  R    2:15:25      1 gpu-node-05
  12348       gpu rl_rl_hy lstudent  R    1:45:10      1 cpu-node-12
  12349       gpu rl_rl_pc lstudent PD       0:00      0 (Resources)
```

The `ST` column shows status: `R` = running, `PD` = pending (waiting for resources).

**Watch the live log output**:
```bash
tail -f outputs/rl_training/20260222_140000/logs/rl_hybrid_hh.log
```

Press Ctrl+C to stop watching the log (this does NOT stop the training job).

**View training metrics with TensorBoard** (optional, requires port forwarding):

On your local machine, set up an SSH tunnel:
```bash
ssh -L 6006:localhost:6006 lstudent@hpc.university.edu
```

Then on the HPC:
```bash
source ~/venvs/rl_packing/bin/activate
tensorboard --logdir outputs/rl_training/20260222_140000/ --port 6006
```

Open your web browser and go to `http://localhost:6006`. You will see live training curves.

### Step 7: Download results to your local machine

After training completes, copy the results back:

```bash
# Run this on your LOCAL machine, not on the HPC
scp -r lstudent@hpc.university.edu:/home/lstudent/bin_packing/outputs/rl_training/ "python/full workflow/outputs/rl_training/"
```

### Step 8: Evaluate and compare all strategies

On the HPC (or locally if you copied the results):

```bash
bash strategies/rl_common/hpc/evaluate_all.sh outputs/rl_training/20260222_140000
```

This runs each trained model on 100 test episodes with the same random seed, then generates comparison plots.

**Expected output**:
```
-- Evaluating: rl_dqn --
  Mean fill: 0.612, Std: 0.054

-- Evaluating: rl_ppo --
  Mean fill: 0.634, Std: 0.048

-- Evaluating: rl_a2c_masked --
  Mean fill: 0.628, Std: 0.051

-- Evaluating: rl_hybrid_hh --
  Mean fill: 0.645, Std: 0.042

-- Evaluating: rl_pct_transformer --
  Mean fill: 0.639, Std: 0.047

-- Generating comparison plots --
  Saved: outputs/rl_training/20260222_140000/evaluation/comparison/fill_rate_comparison.png
  Saved: outputs/rl_training/20260222_140000/evaluation/comparison/box_plot.png
  Saved: outputs/rl_training/20260222_140000/evaluation/comparison/training_curves.png
  Saved: outputs/rl_training/20260222_140000/evaluation/comparison/radar_chart.png
```

---

## 6. Understanding the Output

### 6.1 Files Created During Training

When you train any strategy, a structured set of files is created in the output directory. Here is what each one contains:

```
outputs/rl_hybrid_hh/                         <-- output directory
  checkpoints/
    best_model.pt                             <-- The best model found during training (USE THIS)
    ep_001000.pt                              <-- Checkpoint at episode 1,000
    ep_005000.pt                              <-- Checkpoint at episode 5,000
    ep_010000.pt                              <-- Checkpoint at episode 10,000
    ...                                       <-- More periodic checkpoints
  logs/
    metrics.csv                               <-- All metrics in spreadsheet format
    config.json                               <-- Exact settings used for this training run
    tensorboard/                              <-- TensorBoard event files (for live plots)
      events.out.tfevents.1708617600.gpu03    <-- Binary file, read by TensorBoard
    plots/
      training_curves.png                     <-- Reward and fill rate over episodes
      reward_distribution.png                 <-- Histogram of per-episode rewards
      fill_rate_progress.png                  <-- Fill rate improvement over time
```

**The most important file is `best_model.pt`** -- this is the trained model you will use for evaluation and benchmarking.

### 6.2 How to Read the Training Logs

The `metrics.csv` file contains one row per episode with columns like:

| episode | reward | fill_rate | loss | epsilon | time_s |
|---------|--------|-----------|------|---------|--------|
| 1 | 2.31 | 0.38 | 0.82 | 1.00 | 4.2 |
| 2 | 3.15 | 0.42 | 0.71 | 0.99 | 3.8 |
| ... | ... | ... | ... | ... | ... |
| 10000 | 6.82 | 0.63 | 0.12 | 0.05 | 2.1 |

You can open this file in Excel, Google Sheets, or Python:

```bash
python -c "
import pandas as pd
df = pd.read_csv('outputs/rl_hybrid_hh/logs/metrics.csv')
print('=== Last 5 episodes ===')
print(df.tail())
print()
print('=== Summary ===')
print(df.describe())
"
```

**Expected output**:
```
=== Last 5 episodes ===
       episode  reward  fill_rate    loss  epsilon  time_s
9995      9996    6.21     0.612  0.0821     0.05    2.34
9996      9997    7.03     0.654  0.0793     0.05    2.18
9997      9998    5.89     0.598  0.0845     0.05    2.45
9998      9999    6.45     0.621  0.0812     0.05    2.29
9999     10000    6.78     0.638  0.0798     0.05    2.31

=== Summary ===
          episode     reward  fill_rate      loss   epsilon    time_s
count  10000.000  10000.000  10000.000  10000.000  10000.00  10000.00
mean    5000.500      5.123      0.548      0.312      0.42      2.85
std     2886.896      1.456      0.078      0.241      0.35      0.52
min        1.000      0.821      0.285      0.024      0.05      1.52
max    10000.000      8.234      0.731      0.892      1.00      5.31
```

**What to look for**:
- **reward** should generally increase over episodes (the agent is learning)
- **fill_rate** should generally increase (the agent is packing better)
- **loss** should generally decrease (the neural network's predictions are improving)
- **epsilon** should decrease from 1.0 to 0.05 (the agent gradually shifts from exploring randomly to exploiting what it has learned)

### 6.3 How to Interpret the Plots

The `training_curves.png` plot shows three subplots:

1. **Top: Reward over episodes** -- A line that should trend upward with some noise. The smoothed version (dark line) should clearly go up. If it stays flat, the agent is not learning.

2. **Middle: Fill rate over episodes** -- The metric we care about most. Should trend upward from ~0.35 (random) to 0.55-0.65 (trained).

3. **Bottom: Loss over episodes** -- Should trend downward. If loss starts going up after going down, the agent might be "overfitting" (memorising instead of learning general patterns).

### 6.4 What "Fill Rate" Means

**Fill rate** (also called "volumetric utilisation") is the single most important metric. It answers: "What percentage of the pallet's available volume is actually filled with boxes?"

```
Fill Rate = Total volume of all placed boxes / Total available volume of the pallet
```

**What is a good fill rate?**

| Fill Rate | Rating | Context |
|-----------|--------|---------|
| 30-40% | Poor | Random placement, untrained agent |
| 40-50% | Below average | Early training, simple heuristics |
| 50-60% | Average | Decent heuristic or partially trained RL |
| 60-65% | Good | Well-tuned heuristic (e.g., baseline at 64.8%) |
| 65-70% | Very good | Top heuristic (walle_scoring at 68.3%) or trained RL |
| 70%+ | Excellent | Would be a strong thesis result |

Note: These numbers are for the Botko BV setup with random box distributions. Real-world distributions may yield different numbers.

**Important**: In our multi-pallet setup, the fill rate is calculated only for **closed pallets** (pallets that reached the 1800mm height threshold). Active pallets that are still being packed when the boxes run out are NOT counted. This makes the metric more realistic -- it represents what actually ships out the door.

---

## 7. Troubleshooting FAQ

### "ModuleNotFoundError: No module named 'config'" (or 'simulator', 'strategies')

**What it means**: Python cannot find the project files because you are running the command from the wrong directory.

**Fix**: Make sure you are in the "full workflow" directory, and set the PYTHONPATH:

```bash
cd "python/full workflow"
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

On Windows Git Bash:
```bash
cd "C:/Users/Louis/Downloads/stapelalgortime/python/full workflow"
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

Then try your command again. If you are on HPC, add the export to your SLURM script (train_all.sh already does this).

### "ModuleNotFoundError: No module named 'torch'"

**What it means**: PyTorch is not installed, or you are not in the correct virtual environment.

**Fix**:

If you are on HPC:
```bash
source ~/venvs/rl_packing/bin/activate
python -c "import torch; print(torch.__version__)"
```

If you are on your local machine:
```bash
pip install torch torchvision
```

### "RuntimeError: CUDA out of memory"

**What it means**: The GPU does not have enough memory for the batch size or number of parallel environments you requested.

**Fix**: Reduce the batch size or number of environments:

```bash
# For rl_dqn: reduce batch size
python strategies/rl_dqn/train.py --episodes 50000 --batch_size 128  # was 256

# For rl_ppo or rl_a2c_masked: reduce number of environments
python strategies/rl_ppo/train.py --total_timesteps 5000000 --num_envs 8  # was 16

# For rl_pct_transformer: reduce environments and batch size
python strategies/rl_pct_transformer/train.py --episodes 200000 --num_envs 8
```

### "CUDA not available" (but I have a GPU)

**What it means**: PyTorch was installed without GPU support, or the CUDA drivers are not set up.

**Check your GPU**:
```bash
nvidia-smi
```

If this command works and shows your GPU, reinstall PyTorch with CUDA:
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

If `nvidia-smi` shows "command not found", you either do not have an NVIDIA GPU or need to install NVIDIA drivers. On HPC, try:
```bash
module load CUDA/12.1.1
nvidia-smi
```

### "Permission denied" on HPC

**What it means**: The shell scripts are not marked as executable.

**Fix**:
```bash
chmod +x strategies/rl_common/hpc/*.sh
```

### "No checkpoint found" when running evaluation

**What it means**: The evaluation script cannot find a trained model file.

**Fix**: Check that training completed and produced a checkpoint:

```bash
ls outputs/rl_hybrid_hh/checkpoints/
```

If the directory is empty or does not exist, training either did not complete or saved to a different location. Check your `--output_dir` argument. Common locations:

```bash
# Default locations for each strategy:
ls outputs/rl_dqn/checkpoints/best_network.pt
ls outputs/rl_ppo/logs/checkpoints/best_model.pt
ls outputs/rl_a2c_masked/logs/checkpoints/best_model.pt
ls outputs/rl_hybrid_hh/checkpoints/best_model.pt
ls outputs/rl_pct_transformer/logs/best.pt
```

### "sbatch: command not found"

**What it means**: You are trying to submit a SLURM job but you are either not on an HPC, or SLURM is not installed on your system.

**Fix**: If you are on your local machine and just want to run training locally:

```bash
bash strategies/rl_common/hpc/train_all.sh --local
```

This runs all strategies sequentially on your local machine instead of submitting them to SLURM.

### Training seems stuck (no progress for a long time)

**What it means**: This is likely normal. Early episodes are slow because the agent explores randomly and makes many invalid placements. A few tips:

1. Check that the log file is still being written to:
   ```bash
   ls -la outputs/rl_dqn/logs/metrics.csv
   ```
   If the file size is increasing, training is still running.

2. For rl_dqn, the first 1,000 episodes are pure exploration (epsilon=1.0) and you will NOT see improvement. This is expected.

3. For rl_ppo, initial reward values near zero are normal. The agent needs at least 100,000 timesteps before meaningful learning starts.

### "KeyboardInterrupt" / I accidentally stopped training

**What it means**: You pressed Ctrl+C and stopped the training script.

**Fix**: If checkpoints were saved, you can resume from the last one:

```bash
# For rl_dqn
python strategies/rl_dqn/train.py --resume outputs/rl_dqn/checkpoints/ep_010000.pt --episodes 50000

# For other strategies, check if they support --resume (check each strategy's README)
```

If no checkpoints were saved, you need to restart training from scratch.

---

## 8. Which Strategy Should I Train First?

Use this decision flowchart to decide where to start:

```
START
  |
  v
Do you have a GPU available?
  |
  +-- NO --> Train rl_hybrid_hh (CPU only, 1-4 hours)
  |            This is the thesis's novel contribution.
  |            Start with --mode tabular (1 hour)
  |            Then try --mode dqn (4 hours)
  |
  +-- YES --> How much GPU time do you have?
                |
                +-- Less than 12 hours --> Train rl_hybrid_hh (CPU) + rl_dqn (GPU, 12h)
                |
                +-- 12-24 hours --> Add rl_ppo (16h) or rl_pct_transformer (16h)
                |
                +-- 24+ hours --> Train all 5 strategies in parallel
                |
                +-- Unlimited (HPC) --> Train all 5 with full hyperparameters
```

### Recommended Training Order for Thesis

If you are writing a thesis with these strategies, train them in this order:

1. **rl_hybrid_hh (tabular)** -- Train first because it is fastest (1 hour on CPU). This gives you a baseline RL result immediately. It is also the novel contribution, so you want the most iterations and ablations on this one.

2. **rl_hybrid_hh (DQN)** -- Compare the neural network selector against the tabular Q-table. This comparison (tabular vs. DQN for the same task) is a valuable ablation study for the thesis.

3. **rl_dqn** -- The "standard" deep RL approach. Train this next because it has the most mature codebase (based on well-known algorithms with known performance).

4. **rl_ppo** -- Good comparison point. PPO is the most popular modern RL algorithm, so reviewers will expect to see it.

5. **rl_a2c_masked** -- Interesting for the feasibility masking angle. The CMDP formulation and mask predictor are worth discussing in the thesis even if the final fill rate is similar to PPO.

6. **rl_pct_transformer** -- Train last because it is the most complex and takes the longest. The Transformer architecture is impressive in the thesis but is not necessarily the highest performer.

### Quick Comparison Table

| Strategy | GPU? | Time | Complexity | Thesis Value |
|----------|------|------|------------|--------------|
| rl_hybrid_hh (tabular) | No | ~1h | Low | **Highest** (novel contribution) |
| rl_hybrid_hh (DQN) | No | ~4h | Low | **High** (ablation study) |
| rl_dqn | Yes | ~12h | Medium | High (established baseline) |
| rl_ppo | Yes | ~16h | High | Medium (popular algorithm) |
| rl_a2c_masked | Yes | ~16h | High | Medium (masking analysis) |
| rl_pct_transformer | Yes | ~16h | Very High | Medium (Transformer novelty) |

---

## 9. File Structure Reference

Below is the complete directory tree for all RL-related code, with a one-line description of what each file does.

```
strategies/
|
|-- rl_common/                              Shared infrastructure for all RL strategies
|   |-- __init__.py                         Exports all public classes and functions
|   |-- environment.py                      BinPackingEnv: Gymnasium environment that wraps the
|   |                                         packing simulator so RL agents can interact with it
|   |-- rewards.py                          RewardShaper: computes multi-component reward signals
|   |                                         (volume ratio, surface contact, height penalty, etc.)
|   |-- obs_utils.py                        Observation encoding utilities (heightmap normalisation,
|   |                                         feature extraction, tensor conversion)
|   |-- logger.py                           TrainingLogger: writes metrics to CSV, TensorBoard,
|   |                                         and generates matplotlib training curve plots
|   |-- compare_strategies.py               Generates thesis-quality comparison plots across all
|   |                                         strategies (bar charts, box plots, radar charts)
|   |-- README.md                           Overview of all 5 RL strategies and shared architecture
|   |
|   |-- hpc/                                HPC deployment scripts and configuration
|       |-- requirements.txt                Python dependencies (torch, gymnasium, matplotlib, etc.)
|       |-- setup_hpc.sh                    One-time script to create venv and install everything
|       |-- train_all.sh                    Submits all 5 training jobs to SLURM in parallel
|       |-- evaluate_all.sh                 Evaluates all trained models and generates comparison plots
|       |-- README.md                       HPC-specific quick-start guide
|
|-- rl_dqn/                                 Strategy 1: Double DQN with candidate-based action space
|   |-- __init__.py                         Exports RLDQNStrategy for the strategy registry
|   |-- config.py                           DQNConfig dataclass: all hyperparameters (lr, gamma,
|   |                                         buffer size, epsilon schedule, etc.)
|   |-- network.py                          CNN + MLP Q-network with dueling architecture
|   |                                         (processes heightmaps + box features -> Q-values)
|   |-- replay_buffer.py                    Experience replay buffer with prioritised sampling (PER)
|   |                                         and N-step return computation
|   |-- candidate_generator.py              Generates 50-200 smart placement candidates using
|   |                                         corner alignment, extreme points, EMS, and grid scan
|   |-- train.py                            Full training loop: episodes, epsilon decay, checkpointing,
|   |                                         logging, evaluation intervals
|   |-- evaluate.py                         Evaluates a trained checkpoint on test episodes and
|   |                                         compares against heuristic baselines
|   |-- strategy.py                         RLDQNStrategy: BaseStrategy wrapper that loads a trained
|   |                                         checkpoint and runs inference at ~2-5ms per box
|   |-- README.md                           Detailed documentation with architecture diagrams
|
|-- rl_ppo/                                 Strategy 2: PPO with decomposed action space
|   |-- __init__.py                         Exports RLPPOStrategy for the strategy registry
|   |-- config.py                           PPOConfig dataclass: timesteps, num_envs, clip ratio,
|   |                                         GAE lambda, entropy coefficient, etc.
|   |-- network.py                          ActorCritic network: shared CNN encoder, cross-attention
|   |                                         over bins, 4 decomposed policy heads (bin/x/y/orient)
|   |-- train.py                            Vectorized PPO training with parallel environments,
|   |                                         GAE advantage estimation, mini-batch updates
|   |-- evaluate.py                         Evaluation script with baseline comparison support
|   |-- strategy.py                         RLPPOStrategy: BaseStrategy wrapper for inference
|   |-- README.md                           Full documentation with decomposed action space details
|
|-- rl_a2c_masked/                          Strategy 3: A2C with learned feasibility mask
|   |-- __init__.py                         Exports RLA2CMaskedStrategy for the strategy registry
|   |-- config.py                           A2CMaskedConfig: grid step, mask loss weight,
|   |                                         curriculum phases, infeasibility penalty weight
|   |-- network.py                          Three-headed network: actor (policy) + critic (value)
|   |                                         + mask predictor (P(valid|state,action))
|   |-- train.py                            Training with curriculum learning (easy -> hard) and
|   |                                         BCE-supervised mask predictor
|   |-- evaluate.py                         Evaluation with mask ablation study
|   |                                         (predicted mask vs ground-truth mask comparison)
|   |-- strategy.py                         RLA2CMaskedStrategy: BaseStrategy wrapper for inference
|   |-- README.md                           CMDP formulation, masking theory, curriculum details
|
|-- rl_hybrid_hh/                           Strategy 4: RL hyper-heuristic (NOVEL thesis contribution)
|   |-- __init__.py                         Exports RLHybridHHStrategy for the strategy registry
|   |-- config.py                           HHConfig: mode (tabular/dqn), heuristic portfolio,
|   |                                         feature dimensions, exploration schedule
|   |-- state_features.py                   Extracts 39 handcrafted features from the current state
|   |                                         (per-bin stats, box info, buffer info, progress, history)
|   |-- network.py                          TabularQLearner (Q-table, 45K entries) +
|   |                                         HeuristicSelectorDQN (small MLP, ~27K parameters) +
|   |                                         ReplayBuffer
|   |-- train.py                            Training script supporting both tabular Q-learning
|   |                                         and DQN modes via --mode flag
|   |-- evaluate.py                         Evaluation with interpretability analysis: heuristic
|   |                                         selection distribution, phase-dependent preferences,
|   |                                         Q-value landscape, failure recovery patterns
|   |-- strategy.py                         RLHybridHHStrategy: loads trained model and delegates
|   |                                         placement to the selected heuristic at each step
|   |-- README.md                           Research gap analysis, comparison with prior work,
|   |                                         interpretability discussion
|
|-- rl_pct_transformer/                     Strategy 5: Transformer-based actor-critic (PCT-inspired)
|   |-- __init__.py                         Exports RLPCTTransformerStrategy for the strategy registry
|   |-- config.py                           PCTTransformerConfig: Transformer dimensions, heads,
|   |                                         layers, candidate limits, PPO hyperparameters
|   |-- network.py                          PCTTransformerNet: Transformer encoder (3 layers, 4 heads)
|   |                                         + pointer decoder for variable-size action selection
|   |-- candidate_generator.py              Generates 30-200 validated placement candidates using
|   |                                         corner points, extreme points, floor scan, residual gaps
|   |-- train.py                            PPO training with variable action spaces, padded
|   |                                         mini-batches, and masked softmax
|   |-- evaluate.py                         Evaluation and comparison with heuristic baselines
|   |-- strategy.py                         RLPCTTransformerStrategy: loads Transformer model,
|   |                                         falls back to extreme_points if no checkpoint found
|   |-- README.md                           PCT paper adaptation details, Transformer vs GAT comparison
```

### Quick Reference: Key Commands

| What you want to do | Command |
|---------------------|---------|
| Train rl_hybrid_hh (fast, CPU) | `python strategies/rl_hybrid_hh/train.py --mode tabular --episodes 10000` |
| Train rl_hybrid_hh (DQN, CPU) | `python strategies/rl_hybrid_hh/train.py --mode dqn --episodes 50000` |
| Train rl_dqn (GPU) | `python strategies/rl_dqn/train.py --episodes 50000 --batch_size 256` |
| Train rl_ppo (GPU) | `python strategies/rl_ppo/train.py --total_timesteps 5000000 --num_envs 16` |
| Train rl_a2c_masked (GPU) | `python -m strategies.rl_a2c_masked.train --num_updates 200000 --num_envs 16` |
| Train rl_pct_transformer (GPU) | `python -m strategies.rl_pct_transformer.train --episodes 200000 --num_envs 16` |
| Train all on HPC | `sbatch strategies/rl_common/hpc/train_all.sh` |
| Train all locally | `bash strategies/rl_common/hpc/train_all.sh --local` |
| Evaluate one strategy | `python strategies/rl_hybrid_hh/evaluate.py --checkpoint <path_to_best_model.pt>` |
| Evaluate all strategies | `bash strategies/rl_common/hpc/evaluate_all.sh <training_output_dir>` |
| Compare strategies (plots) | `python strategies/rl_common/compare_strategies.py --eval_dir <eval_dir>` |
| Monitor HPC jobs | `squeue -u $USER` |
| Watch training log | `tail -f outputs/rl_training/*/logs/*.log` |
| View TensorBoard | `tensorboard --logdir outputs/rl_training/` |

---

**You are now ready to train your first RL strategy.** Start with Section 4 (Quick Start) to get something running in 5 minutes, then move to Section 5 (HPC) when you are ready for full-scale training. Good luck!
