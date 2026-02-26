import sys
print("1")
from config import Box, ExperimentConfig, Orientation
print("2")
from simulator.bin_state import BinState
print("3")
from strategies.base_strategy import MultiBinStrategy, MultiBinDecision, register_multibin_strategy
print("4")
import strategies.two_bounded_best_fit.strategy
print("5")
