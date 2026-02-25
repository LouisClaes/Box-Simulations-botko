"""
Quick test to verify the custom close policy closes pallets correctly.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Box, BinConfig
from dataset.generator import generate_rajapack
from strategies.base_strategy import get_strategy
from simulator.session import PackingSession, SessionConfig
from simulator.close_policy_custom import FullestOnConsecutiveRejectsPolicy

# Test configuration: same as overnight runner
BOTKO_PALLET = BinConfig(
    length=1200.0, width=800.0, height=2700.0, resolution=10.0,
)

test_config = SessionConfig(
    bin_config=BOTKO_PALLET,
    num_bins=2,
    buffer_size=8,
    pick_window=4,
    close_policy=FullestOnConsecutiveRejectsPolicy(
        max_consecutive=4,
        min_fill_to_close=0.3  # Lower threshold for testing
    ),
    max_consecutive_rejects=50,  # Allow more rejects for testing
    enable_stability=False,
    allow_all_orientations=False,
)

# Generate MANY boxes to force the pallets to fill up and get stuck
boxes = generate_rajapack(n=400, seed=42)

# Run with a simple strategy
strategy = get_strategy("baseline")
session = PackingSession(test_config)

print(f"Testing close policy with 400 boxes, 2 pallets...")
print(f"Close policy: {test_config.close_policy.describe()}")
print()

result = session.run(boxes, strategy)

print(f"Results:")
print(f"  Boxes placed: {result.total_placed}")
print(f"  Boxes rejected: {result.total_rejected}")
print(f"  Pallets closed: {len(result.closed_pallets)}")
print()

if len(result.closed_pallets) > 0:
    print("✅ SUCCESS - Pallets are closing!")
    for i, pallet in enumerate(result.closed_pallets):
        print(f"  Pallet {i+1}: {pallet.boxes_placed} boxes, {pallet.fill_rate:.1%} fill")
else:
    print("❌ FAILURE - No pallets closed")
    print(f"  Active pallets: {len(result.active_pallets)}")
    for i, pallet in enumerate(result.active_pallets):
        print(f"    Pallet {i+1}: {pallet.boxes_placed} boxes, {pallet.fill_rate:.1%} fill")
