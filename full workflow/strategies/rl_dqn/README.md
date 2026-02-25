# rl_dqn — Double DQN for 3D Online Bin Packing

A production-quality Double DQN reinforcement learning strategy for the
Botko BV dual-pallet bin packing problem.

## How This Strategy Works (Beginner Guide)

This section explains the strategy from the ground up, with no machine learning
background assumed.

### What is a Q-value?

Imagine you are playing a board game and you keep a personal score card for every
move you could make in every situation. Before making a move, you look up its
score on your card and pick the move with the highest number.

That score is called a **Q-value** (short for "quality value"). It answers the
question: "If I am in this situation and I take this action, how much total reward
can I expect to earn from now until the end of the game?"

In bin packing, each "move" is placing the current box at a specific position,
in a specific orientation, on a specific pallet. The Q-value for that move
estimates how much total pallet fill we can eventually achieve if we make that
move and then continue playing well afterwards.

- High Q-value = good move, expect a well-packed pallet in the end
- Low Q-value = bad move, the pallet will likely end up with lots of empty space

During training, the network updates its Q-values after each placement by
comparing what it predicted to what actually happened.

### What is "Double" DQN?

The original DQN algorithm has a known problem: it tends to be overconfident.
When it estimates Q-values, it often makes them too high, which causes it to
keep choosing moves that look great on paper but are actually mediocre.

Think of it like asking one person to both recommend a restaurant AND rate how
good that restaurant is. They are likely to recommend their favourite and then
give it an unrealistically high rating to justify the choice.

**Double DQN** fixes this by using two separate networks:

1. **The online network** (the student): makes the decision about WHICH position
   to place the box. It answers "which move looks best?"
2. **The target network** (the examiner): independently evaluates HOW GOOD that
   chosen move actually is. It answers "given that move, what score should it get?"

The target network is a slightly older copy of the online network that gets
updated only every 500 steps. Because they are not identical, they are less
likely to make the same overestimation errors. This two-network separation
leads to more accurate Q-values and more reliable learning.

### What is "Prioritised Experience Replay"?

During training, every decision the agent makes (the situation it was in, the
move it chose, the reward it got, the new situation it ended up in) is saved in
a large memory bank called a **replay buffer** (this one holds 500,000 past
experiences).

When it is time to train, the network picks a random batch of past experiences
from this buffer and learns from them, just like a student reviewing flashcards.

**The problem with purely random review**: if the student picks flashcards
completely at random, they will keep reviewing the easy ones they already know,
wasting time on things they have mastered.

**Prioritised Experience Replay** solves this by assigning each experience a
priority score based on how surprised the network was by the outcome. Experiences
where the network predicted poorly (big prediction error = big "surprise") get a
higher priority and are reviewed more often. Experiences the network already
handles well get reviewed less.

The result: the network focuses its training time on the situations it finds
hardest, which makes learning faster and more efficient.

### What are "candidates"?

The pallet grid is 1200mm x 800mm with a resolution of 10mm, which means there
are 120 x 80 = 9,600 grid positions per bin. With 2 bins and 2 orientations
(rotate the box 90 degrees or not), the total number of possible moves is:

    120 x 80 x 2 x 2 = 38,400 possible actions per step

Evaluating all 38,400 positions for every single box placement would be very
slow. More importantly, most of those positions are geometrically terrible (the
box would float in mid-air, or clip through an existing box).

**Candidates** are a smart shortcut. Instead of checking all 38,400 positions,
the system first uses geometric rules to identify only the 50-200 most promising
positions:

- **Corner points**: the corners of already-placed boxes are natural spots to
  stack the next box flush against
- **Extreme points**: positions at the edge of height changes in the heightmap
  (think of the edge of a shelf)
- **EMS-inspired positions**: corners of the largest empty rectangular spaces
  inside the bin
- **Coarse grid fallback**: a sparse sample of grid positions as a safety net

Only these candidates are sent to the neural network for Q-value scoring. This
makes the system 100-200x faster with minimal loss in packing quality.

### Step-by-step: what happens during one training step

1. **Observe the state**: the network reads the current heightmap of both pallets
   (a top-down height image, like a topographic map) and the features of the
   next 4 boxes visible on the conveyor belt.

2. **Generate candidates**: the candidate generator proposes 50-200 promising
   positions for the current box.

3. **Choose a move (epsilon-greedy)**: with probability epsilon (which starts at
   1.0 and slowly falls to 0.05 over training), the agent picks a random
   candidate to explore. Otherwise it picks the candidate with the highest
   Q-value. Early in training, almost all moves are random (exploration). Late
   in training, almost all moves use the learned Q-values (exploitation).

4. **Execute the move**: the box is placed on the pallet. The simulator checks
   if the placement is valid and returns a reward (based on volume ratio and fill
   improvement) and the new pallet state.

5. **Store the experience**: the (state, action, reward, next state) tuple is
   stored in the prioritised replay buffer with a high initial priority.

6. **Sample and learn**: a batch of 256 high-priority past experiences is
   sampled. For each experience:
   - The online network picks the best next action
   - The target network evaluates that action's Q-value
   - The loss is the squared difference between predicted and target Q-value
     (using Huber loss, which is robust to large errors)
   - Gradients are clipped to prevent unstable updates

7. **Update priorities**: the buffer priorities are updated based on how
   surprising each experience turned out to be.

8. **Periodically sync the target network**: every 500 steps, the target network
   gets updated to match the online network.

### Step-by-step: what happens during inference (placing one box)

At inference time (when the trained model is actually packing boxes), the process
is simpler:

1. **Observe**: read the current heightmaps and box features.
2. **Generate candidates**: the same geometric rules produce 50-200 positions.
3. **Score**: the neural network computes a Q-value for every candidate in one
   forward pass (this takes approximately 2-5 milliseconds).
4. **Pick the best**: the candidate with the highest Q-value is selected.
5. **Place**: the box is placed at that position.

There is no randomness at inference time. The network always picks its best
known move.

---

## Algorithm Overview

This implementation extends the classical DQN (Mnih et al. 2015) with
three critical improvements for the bin packing domain:

1. **Double DQN** (van Hasselt et al. 2016): Decouples action selection
   (online network) from value estimation (target network) to reduce
   overestimation bias.

2. **Candidate-based action space**: Instead of evaluating all 38,400
   grid positions, generates 50-200 smart candidates using corner alignment,
   extreme points, EMS-inspired positions, and a coarse grid fallback.
   This makes training 100-200x faster with minimal quality loss.

3. **Dueling architecture** (Wang et al. 2016): Separates state value V(s)
   from action advantage A(s,a), enabling better generalisation when many
   actions have similar values (common in bin packing).

Additionally:
- Prioritised experience replay (Schaul et al. 2016)
- N-step returns (n=3) for faster credit assignment
- Huber loss with gradient clipping for training stability

## Architecture

```
                 +-------------------+
                 | Heightmaps (2ch)  |   (batch, 2, 120, 80)
                 |   Bin 1 + Bin 2   |
                 +--------+----------+
                          |
                   CNN Branch (4 conv layers)
                   Conv(2,32,5,s=2) -> BN -> ReLU
                   Conv(32,64,3,s=2) -> BN -> ReLU
                   Conv(64,128,3,s=1) -> BN -> ReLU
                   Conv(128,256,3,s=1) -> BN -> ReLU
                   GlobalAvgPool -> 256-dim
                          |
                          v
+-------------------+  +-----+  +-------------------+
| Box Features (20) |  | 256 |  | Action Features(7)|
| pick_window x 5   |  +--+--+  | bin,x,y,o,z,sup,h|
+--------+----------+     |     +--------+----------+
         |                 |              |
    Box MLP               State        Act MLP
    20->128->128         Embed        7->64->64
         |               (384)             |
         v                 |               v
      +--+--+              |           +---+---+
      | 128 |              |           |  64   |
      +--+--+              |           +---+---+
         |                 |               |
         +--------+--------+               |
                  |                         |
               +--v--+                      |
               | 384 |                      |
               +--+--+                      |
                  |                         |
     +------------+-------------------------+
     |            Dueling Architecture      |
     |                                      |
     v                                      v
  V(s) stream                     A(s,a) stream
  384->256->128->1                448->256->128->1
     |                                      |
     +----------> Q = V + A - mean(A) <-----+
                        |
                     Q-value (scalar)
```

## File Structure

```
strategies/rl_dqn/
    __init__.py            Import RLDQNStrategy
    config.py              All hyperparameters (DQNConfig dataclass)
    network.py             PyTorch network (CNN + MLP branches + dueling)
    replay_buffer.py       Uniform + PER buffers + n-step wrapper
    candidate_generator.py Smart candidate positions (corner/EP/EMS/grid)
    strategy.py            RLDQNStrategy (BaseStrategy for inference)
    train.py               Full training script with CLI
    evaluate.py            Evaluation + baseline comparison
    README.md              This file
```

## Training

### Local (development)

```bash
cd "python/full workflow"

# Quick test (100 episodes)
python strategies/rl_dqn/train.py --episodes 100 --eval_interval 50 --log_interval 10

# Full training
python strategies/rl_dqn/train.py --episodes 50000 --batch_size 256 --lr 0.001 --gamma 0.95

# Resume from checkpoint
python strategies/rl_dqn/train.py --resume outputs/rl_dqn/checkpoints/ep_010000.pt
```

### HPC (production)

Transfer these directories to the HPC node:
- `strategies/` (all strategy code + rl_common + rl_dqn)
- `simulator/` (PackingSession, BinState, etc.)
- `config.py` (data models)

```bash
# SLURM example
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

python strategies/rl_dqn/train.py \
    --episodes 50000 \
    --batch_size 256 \
    --lr 0.001 \
    --output_dir /scratch/$USER/rl_dqn
```

### Monitoring

```bash
# TensorBoard (local)
tensorboard --logdir outputs/rl_dqn/logs/tensorboard

# CSV analysis
python -c "import pandas as pd; df = pd.read_csv('outputs/rl_dqn/logs/metrics.csv'); print(df.tail())"
```

## Evaluation

```bash
# Basic evaluation
python strategies/rl_dqn/evaluate.py \
    --checkpoint outputs/rl_dqn/checkpoints/best_network.pt \
    --episodes 50

# With baseline comparison
python strategies/rl_dqn/evaluate.py \
    --checkpoint outputs/rl_dqn/checkpoints/best_network.pt \
    --episodes 100 \
    --compare baseline walle_scoring surface_contact
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 0.001 | Adam learning rate |
| `batch_size` | 256 | Minibatch size |
| `gamma` | 0.95 | Discount factor |
| `eps_start` | 1.0 | Initial epsilon |
| `eps_end` | 0.05 | Final epsilon |
| `eps_decay_fraction` | 0.8 | Fraction of episodes for decay |
| `buffer_capacity` | 500,000 | Replay buffer size |
| `buffer_alpha` | 0.6 | PER priority exponent |
| `n_step` | 3 | N-step return horizon |
| `target_update_freq` | 500 | Steps between target syncs |
| `max_candidates` | 200 | Max candidates per step |
| `use_dueling` | True | Dueling architecture |
| `use_batch_norm` | True | Batch norm in CNN |
| `grad_clip` | 10.0 | Max gradient norm |

## Expected Results

After 50,000 episodes of training:
- Fill rate: 55-65% (random boxes, online setting)
- Competitive with top heuristics (walle_scoring, surface_contact)
- Speed: ~2-5ms per box decision at inference

Note: Results depend heavily on the box size distribution. With Botko BV's
real distribution (rather than uniform random), results may differ.

## Integration

The strategy is registered as `"rl_dqn"` in the strategy registry:

```python
from strategies.base_strategy import get_strategy
strategy = get_strategy("rl_dqn")  # loads default checkpoint
result = session.run(boxes, strategy=strategy)
```

For the overnight Botko experiment:
```python
strategy = RLDQNStrategy(checkpoint_path="path/to/best.pt")
result = session.run(boxes, strategy=strategy, box_selector=FIFOBoxSelector())
```

## References

1. Mnih, V., et al. (2015). Human-level control through deep reinforcement
   learning. *Nature*, 518(7540), 529-533.

2. van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning
   with Double Q-learning. *AAAI*, 2094-2100.

3. Wang, Z., et al. (2016). Dueling network architectures for deep reinforcement
   learning. *ICML*, 1995-2003.

4. Schaul, T., et al. (2016). Prioritized experience replay. *ICLR*.

5. Tsang, Y., et al. (2025). Deep reinforcement learning for online 3D bin
   packing with dual bins. *SIMPAC*, 311.

6. Zhao, H., et al. (2021). Online 3D bin packing with constrained deep
   reinforcement learning. *AAAI*, 741-749.

7. Xiong, L., et al. (2024). GOPT: Generalizable online 3D bin packing via
   transformer-based deep reinforcement learning. *RA-L*.
