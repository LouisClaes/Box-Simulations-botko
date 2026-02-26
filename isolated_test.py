import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Box, BinConfig
from simulator.bin_state import BinState
from strategies.two_bounded_best_fit.strategy import TwoBoundedBestFitStrategy

print("Starting isolated Best-In-Bin test...")
try:
    cfg = BinConfig(length=1200.0, width=800.0, height=2700.0, resolution=10.0)
    bstate = BinState(cfg)
    strategy = TwoBoundedBestFitStrategy()
    strategy.on_episode_start([cfg, cfg])
    
    box = Box(id="test1", length=100.0, width=100.0, height=100.0)
    
    print("Evaluating decision...")
    t0 = time.time()
    decision = strategy.decide_placement(box, [bstate])
    t1 = time.time()
    
    print(f"Decision: {decision}")
    print(f"Time taken: {t1-t0:.4f}s")
except Exception as e:
    import traceback
    traceback.print_exc()

print("Done.")
