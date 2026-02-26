import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Box, BinConfig
from simulator.session import PackingSession, SessionConfig
from simulator.close_policy import HeightClosePolicy
from strategies.base_strategy import get_multibin_strategy
from dataset.generator import generate_rajapack

print("Validating Two-Bounded Best Fit against PRD Metrics...")
print("Goal: Placement Rate > 85%, Closed Fill > 65%")

# Exact standard botko config
BOTKO_PALLET = BinConfig(length=1200.0, width=800.0, height=2700.0, resolution=10.0)
BOTKO_SESSION_CONFIG = SessionConfig(
    bin_config=BOTKO_PALLET,
    num_bins=2,
    buffer_size=8,
    pick_window=4,
    close_policy=HeightClosePolicy(max_height=1800.0),
    max_consecutive_rejects=10,
    enable_stability=False,
    allow_all_orientations=False,
)

if __name__ == '__main__':
    metrics_placement = []
    metrics_fill = []
    
    for i in range(5):
        t0 = time.time()
        strategy = get_multibin_strategy("two_bounded_best_fit")
        session = PackingSession(BOTKO_SESSION_CONFIG)
        boxes = generate_rajapack(100, seed=42 + i)
        
        res = session.run(boxes, strategy)
        
        pr = res.total_placed / 100.0 * 100.0
        cfr_val = res.avg_closed_fill * 100.0
        
        pr_str = f"{pr:.1f}%"
        cfr_str = f"{cfr_val:.1f}%" if cfr_val > 0 else "0.0% (No closed pallets)"
        
        print(f"Dataset {i+1}/5:")
        print(f"  Placement Rate:  {pr_str}")
        print(f"  Closed Fill:     {cfr_str}")
        print(f"  Total Rejected:  {res.total_rejected}")
        print(f"  Runtime:         {time.time()-t0:.1f}s")
        
        metrics_placement.append(pr)
        if cfr_val > 0:
            metrics_fill.append(cfr_val)
            
    avg_pr = sum(metrics_placement) / len(metrics_placement)
    print(f"\n--- OVERALL AVERAGES ---")
    print(f"Placement Rate: {avg_pr:.1f}% (Target: >85%)")
    if metrics_fill:
        avg_fr = sum(metrics_fill) / len(metrics_fill)
        print(f"Closed Fill:    {avg_fr:.1f}% (Target: >65%)")
    else:
        print("Closed Fill:    0.0% (Warning: Never hit 50% to close)")
