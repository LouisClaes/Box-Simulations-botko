# Tsang Multi-Bin Strategy

## Core Logic
`tsang_multibin` is a native `MultiBinStrategy` for `MultiBinPipeline`.

For each bin, it runs a strict candidate pass and, if needed, a denser fallback pass. The best in-bin placement is then combined with a height-tapered best-fit bin bonus.

### In-bin score
For a feasible candidate `(x, y, z, ol, ow, oh)`:

```text
height_norm   = z / H
top_norm      = (z + oh) / H
height_growth = max(0, (z + oh - current_max_h) / H)
position_pen  = (x + y) / (L + W)
tower_pen     = top_norm^2 + height_growth * (1 - contact_ratio)

score = + 6.0 * contact_ratio
        + 1.5 * support_ratio
        - 3.0 * height_norm
        - 7.0 * height_growth
        - 5.0 * tower_pen
        - 0.5 * position_pen
```

### Cross-bin routing score
```text
height_taper = max(0, 1 - max_height / (0.72 * H))
fill_bonus   = fill_rate * 2.0 * height_taper

total_score  = in_bin_score + fill_bonus
```

This keeps best-fit behavior while reducing late-stage tall-tower bias.

## Candidate Generation
Primary and fallback passes both use margin-aware candidates:
- legal wall anchors (`margin`-compliant),
- placed-box corner/edge anchors,
- orientation-offset anchors (`p.x - ol - margin`, `p.y - ow - margin`, etc.),
- interior grid sweep.

Fallback pass adds denser local jitter around recent anchors and a tighter grid step.

## Feasibility Rules
All phases keep hard constraints strict:
- anti-float support `>= 0.30`,
- optional stability threshold if enabled,
- `is_margin_clear(...)`,
- bin bounds and height limit.

## Quick Validation
From `python/`:

```powershell
python -m compileall strategies/tsang_multibin/strategy.py
```

```powershell
@'
import importlib.util
import pathlib
import sys
import types
from config import Box, BinConfig
from simulator.buffer import BufferPolicy
from simulator.multi_bin_pipeline import MultiBinPipeline, PipelineConfig

root = pathlib.Path.cwd()
pkg = types.ModuleType("strategies")
pkg.__path__ = [str((root / "strategies").resolve())]
sys.modules["strategies"] = pkg

base_path = root / "strategies" / "base_strategy.py"
base_spec = importlib.util.spec_from_file_location("strategies.base_strategy", base_path)
base_mod = importlib.util.module_from_spec(base_spec)
sys.modules["strategies.base_strategy"] = base_mod
base_spec.loader.exec_module(base_mod)

tsang_path = root / "strategies" / "tsang_multibin" / "strategy.py"
tsang_spec = importlib.util.spec_from_file_location("strategies.tsang_multibin.strategy", tsang_path)
tsang_mod = importlib.util.module_from_spec(tsang_spec)
sys.modules["strategies.tsang_multibin.strategy"] = tsang_mod
tsang_spec.loader.exec_module(tsang_mod)

boxes = [
    Box(id=1, length=380, width=280, height=220),
    Box(id=2, length=420, width=320, height=240),
    Box(id=3, length=360, width=260, height=200),
    Box(id=4, length=300, width=240, height=260),
    Box(id=5, length=340, width=220, height=210),
    Box(id=6, length=260, width=200, height=180),
]

cfg = PipelineConfig(
    n_bins=2,
    buffer_size=4,
    buffer_policy=BufferPolicy.LARGEST_FIRST,
    bin_config=BinConfig(length=1200, width=800, height=2700, resolution=10.0, margin=20.0),
)

result = MultiBinPipeline(strategy=tsang_mod.TsangMultiBinStrategy(), config=cfg).run(boxes)
print("placed", result.total_placed, "rejected", result.total_rejected, "fill", round(result.aggregate_fill_rate, 4))
'@ | python -
```
