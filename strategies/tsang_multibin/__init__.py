"""
Tsang Multi-Bin strategy (MultiBinStrategy).

Based on: Tsang et al. (2025), "A deep reinforcement learning approach for
online and concurrent 3D bin packing optimisation with bin replacement strategies",
Computers in Industry, Vol. 164, Article 104202.
GitHub: https://github.com/SoftwareImpacts/SIMPAC-2024-311

Implements the dual-bin packing logic with bin replacement strategies
(FILL, HEIGHT, FAIL, COMBINED) as described in Tsang et al. 2025.
This is the heuristic (non-DRL) version that uses surface contact scoring.
"""
from strategies.tsang_multibin.strategy import TsangMultiBinStrategy
__all__ = ["TsangMultiBinStrategy"]
