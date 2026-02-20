# Heuristic 160 Strategy

## Overview
Implements the 160-Heuristic Framework for 3D Bin Packing from Ali et al. (2024/2025). Enumerates 8 × 5 = 40 placement rule combinations using Empty Maximal Spaces (EMS) as candidate positions.

## Papers
- Ali, I., Ramos, A.G., Carravilla, M.A., Oliveira, J.F. (2024). "A matheuristic for the online 3D bin packing problem in the slaughterhouse industry." *Applied Soft Computing*, Vol. 151, Article 111168. https://doi.org/10.1016/j.asoc.2023.111168
- Ali, I., Ramos, A.G., Oliveira, J.F. (2025). "A 2-phase matheuristic for the online 3D bin packing problem." *Computers & Operations Research*, Vol. 178, Article 107005. https://doi.org/10.1016/j.cor.2025.107005

## Algorithm
1. Rebuild Empty Maximal Spaces (EMS) from all placed boxes using Lai & Chan (1997) split-and-eliminate.
2. Sort EMSs by the configured **space-selection rule** (1–8).
3. For each EMS, rank orientations by the configured **orientation-selection rule** (1–5).
4. Place box at the first valid EMS×orientation combination (first-valid-fit).
5. Fallback: coarse bottom-left-fill grid scan if EMS list yields no valid placement.

**Default configuration: A53** (space rule 5 = DBLF+corner, orient rule 3 = largest base + max x-extent).

### Space-selection rules (8)
| Rule | Description |
|------|-------------|
| 1 | DBLF — leftmost, lowest, frontmost |
| 2 | Lowest first |
| 3 | Lowest-front-left |
| 4 | Smallest EMS volume first |
| **5** | **DBLF + corner preference (default)** |
| 6 | Rightmost first |
| 7 | Min-z only |
| 8 | DFTRC — deep-front-top-right corner |

### Orientation-selection rules (5)
| Rule | Description |
|------|-------------|
| 1 | Min margin (minimise wasted EMS) |
| 2 | Largest base area |
| **3** | **Largest base + greatest x-extent (default)** |
| 4 | Best box/EMS volume ratio |
| 5 | Shortest height + largest base |

## Performance (50 boxes, 1200×800×2700mm EUR pallet, seed=42)
| Metric | Value |
|--------|-------|
| Fill rate | **55.2%** |
| Boxes placed | 36/50 (72%) |
| Computation time | 3.5s total (0.07s/box) |

**Placement rate note:** 28% rejection is expected — EMS fragmentation means later boxes may not find valid EMS slots even when physical space exists. The fallback grid scan helps but is coarse.

## Class: `Heuristic160Strategy(BaseStrategy)`
- **Registry name:** `heuristic_160`
- **Constructor:** `Heuristic160Strategy(space_rule=5, orient_rule=3)`
- **Factory:** `Heuristic160Strategy.make(space_rule, orient_rule)` — create any of the 40 variants

## Stability
- `MIN_SUPPORT = 0.30` always enforced (anti-float)
- Respects `ExperimentConfig.enable_stability` for stricter checks

## Integration
```python
from run_experiment import run_experiment
from config import ExperimentConfig, BinConfig

config = ExperimentConfig(
    strategy_name="heuristic_160",
    bin=BinConfig(length=1200, width=800, height=2700, resolution=10.0),
)
result = run_experiment(config, boxes)
```
