# Two-Bounded Best-Fit Strategy

## Overview
A `MultiBinStrategy` that natively manages two bins simultaneously, selecting the best (bin, position, orientation) combination at each step using surface-contact scoring with a Best-Fit bin preference. Implements the dual-bin bin-routing principles from Zhao et al. (AAAI 2021) and Tsang et al. (2025).

## Papers
- Zhao, H., She, Q., Zhu, C., Yang, Y., & Xu, K. (2021). "Online 3D Bin Packing with Constrained Deep Reinforcement Learning." *AAAI Conference on Artificial Intelligence*, 35(6), 741–749. arXiv:2012.04412. https://github.com/alexfrom0815/Online-3D-BPP-DRL
- Tsang, C.W., Tsang, E.C.C., & Wang, X. (2025). "A deep reinforcement learning approach for online and concurrent 3D bin packing optimisation with bin replacement strategies." *Computers in Industry*, Vol. 164, Article 104202.

## Algorithm
For each incoming box:
1. For each bin: find best (x, y, orientation) using composite surface-contact score.
2. Apply Best-Fit bin preference: bonus proportional to current fill rate.
3. Route box to the globally best (bin, position, orientation).

### Per-bin placement score
```
score = WEIGHT_SUPPORT × support_ratio
       - WEIGHT_HEIGHT × height_norm
       + WEIGHT_CONTACT × contact_base_ratio

total = score + bin_fill_rate × FILL_BONUS_WEIGHT
```

### Key constants
| Constant | Value | Role |
|----------|-------|------|
| `WEIGHT_SUPPORT` | 5.0 | Stability |
| `WEIGHT_HEIGHT` | 2.0 | Vertical efficiency |
| `WEIGHT_CONTACT` | 3.0 | Surface flatness |
| `FILL_BONUS_WEIGHT` | 5.0 | Best-Fit bias (high: strongly prefers fuller bin) |
| `MIN_SUPPORT` | 0.30 | Anti-float |

### Difference from `tsang_multibin`
- `two_bounded_best_fit`: higher FILL_BONUS (5.0 vs 2.0) → more aggressive best-fit behavior; scoring uses 3 components (support + height + contact)
- `tsang_multibin`: lower FILL_BONUS (2.0); scoring uses contact + height + small support bonus

## Performance (50 boxes, 1200×800×2700mm EUR pallet, 2 bins, buffer=5, seed=42)
| Metric | Value |
|--------|-------|
| Aggregate fill rate | **38.3%** (across 2 bins) |
| Boxes placed | 49/50 (98%) |
| Bins used | 2 |
| Computation time | ~77s total (1.57s/box) |

## Class: `TwoBoundedBestFitStrategy(MultiBinStrategy)`
- **Registry name:** `two_bounded_best_fit`
- **Type:** Multi-bin (`MultiBinStrategy`) — requires `MultiBinPipeline`
- **NOT compatible** with single-bin `run_experiment.py` or `MultiBinOrchestrator`

## Integration
```python
from simulator.multi_bin_pipeline import MultiBinPipeline, PipelineConfig
from simulator.buffer import BufferPolicy
from config import BinConfig
import strategies

from strategies.base_strategy import MULTIBIN_STRATEGY_REGISTRY
strategy = MULTIBIN_STRATEGY_REGISTRY["two_bounded_best_fit"]()

config = PipelineConfig(
    n_bins=2,
    buffer_size=5,
    buffer_policy=BufferPolicy.LARGEST_FIRST,
    bin_config=BinConfig(length=1200, width=800, height=2700, resolution=10.0),
)
pipeline = MultiBinPipeline(strategy=strategy, config=config)
result = pipeline.run(boxes)
```

## "2K-Bounded" interpretation
The "two-bounded" name reflects the theoretical 2K-competitive ratio bound for online bin packing with 2 active bins (Zhao et al. 2021 and classical BPP theory): using 2 bins reduces the competitive ratio from ∞ (single bin, worst case) to a bounded factor K.
