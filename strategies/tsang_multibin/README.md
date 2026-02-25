# Tsang Multi-Bin Strategy

## Overview
Heuristic implementation of the dual-bin packing logic from Tsang et al. (2025, Computers in Industry). Implements the bin-routing and bin-replacement mechanics from the DeepPack3D system as a `MultiBinStrategy` for the `MultiBinPipeline`. Routes each box to the bin where it achieves the best surface contact, with a Best-Fit bias (prefer the fuller bin).

## Paper
Tsang, C.W., Tsang, E.C.C., & Wang, X. (2025). "A deep reinforcement learning approach for online and concurrent 3D bin packing optimisation with bin replacement strategies." *Computers in Industry*, Vol. 164, Article 104202. https://doi.org/10.1016/j.compind.2024.104202

GitHub: https://github.com/SoftwareImpacts/SIMPAC-2024-311 (DeepPack3D, MIT license)

## Algorithm
For each incoming box:
1. **Per-bin search**: for each active bin, find the best (x, y, orientation) using surface-contact scoring.
2. **Bin scoring**: apply a Best-Fit fill bonus (prefer the fuller bin).
3. **Cross-bin decision**: route to the globally best (bin, position, orientation).

### Placement score per bin
```
in_bin_score = CONTACT_WEIGHT × contact_ratio
             - HEIGHT_PENALTY_WEIGHT × height_norm
             + 0.5 × support_ratio

fill_bonus = bin_fill_rate × FILL_BONUS_WEIGHT

total_score = in_bin_score + fill_bonus
```

### Key constants
| Constant | Value | Role |
|----------|-------|------|
| `CONTACT_WEIGHT` | 5.0 | Surface contact |
| `FILL_BONUS_WEIGHT` | 2.0 | Best-Fit bias |
| `HEIGHT_PENALTY_WEIGHT` | 1.0 | Prefer low placements |
| `MIN_SUPPORT` | 0.30 | Anti-float |

### Bin replacement strategies (Tsang 2025)
The paper defines 5 replacement policies. The strategy focuses on the **bin-routing** component; replacement is handled externally by the `MultiBinOrchestrator` or `MultiBinPipeline`. The five policies from the paper are:
- **FILL**: Close when utilisation > 85%
- **HEIGHT**: Close when max height > 90%
- **FAIL**: Close after N consecutive rejections
- **COMBINED** (recommended): FILL OR HEIGHT OR FAIL
- **MANUAL**: External trigger

## Performance (50 boxes, 1200×800×2700mm EUR pallet, 2 bins, buffer=5, seed=42)
| Metric | Value |
|--------|-------|
| Aggregate fill rate | **39.2%** (across 2 bins) |
| Boxes placed | 50/50 (100%) |
| Bins used | 2 |
| Computation time | ~92s total (1.84s/box) |

**Note:** 100% placement rate — the dual-bin system eliminates rejections by always routing to the better bin. The aggregate fill (39.2%) reflects 2 bins used for 50 boxes where 1 bin could hold ~65-70%.

## Class: `TsangMultiBinStrategy(MultiBinStrategy)`
- **Registry name:** `tsang_multibin`
- **Type:** Multi-bin (`MultiBinStrategy`) — requires `MultiBinPipeline`
- **NOT compatible** with single-bin `run_experiment.py` or `MultiBinOrchestrator`

## Integration
```python
from simulator.multi_bin_pipeline import MultiBinPipeline, PipelineConfig
from simulator.buffer import BufferPolicy
from config import BinConfig
import strategies  # registers all strategies

from strategies.base_strategy import MULTIBIN_STRATEGY_REGISTRY
strategy = MULTIBIN_STRATEGY_REGISTRY["tsang_multibin"]()

config = PipelineConfig(
    n_bins=2,
    buffer_size=5,
    buffer_policy=BufferPolicy.LARGEST_FIRST,
    bin_config=BinConfig(length=1200, width=800, height=2700, resolution=10.0),
)
pipeline = MultiBinPipeline(strategy=strategy, config=config)
result = pipeline.run(boxes)
print(f"Aggregate fill: {result.aggregate_fill_rate:.1%}")
```

## Comparison with DRL version
This is the **heuristic baseline** for DeepPack3D. The full Tsang 2025 system uses a DQN policy (TensorFlow 2.10) trained with bin replacement strategies. Expected DRL performance: 76.8–79.7% fill. See `python/external_repos/SIMPAC-2024-311/` for the TF implementation (PyTorch port pending).
