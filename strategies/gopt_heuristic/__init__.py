"""
GOPT Corner Heuristic strategy.

Derived from: Xiong et al. (2024), "GOPT: Generalizable Online 3D Bin Packing
via Transformer-based Deep Reinforcement Learning", IEEE RA-L 2024.
GitHub: https://github.com/[GOPT repo]

The heuristic extracts the heightmap corner-detection logic from the GOPT
environment (envs/Packing/ems.py) and uses it to generate placement candidates,
then scores them with a DBLF-like metric.
"""
from strategies.gopt_heuristic.strategy import GOPTHeuristicStrategy
__all__ = ["GOPTHeuristicStrategy"]
