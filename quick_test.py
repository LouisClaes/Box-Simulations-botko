"""Quick smoke test for close policy."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import BinConfig
from dataset.generator import generate_rajapack
from strategies.base_strategy import get_strategy
from simulator.session import PackingSession, SessionConfig
from simulator.close_policy_custom import FullestOnConsecutiveRejectsPolicy

test_config = SessionConfig(
    bin_config=BinConfig(length=1200.0, width=800.0, height=2700.0, resolution=10.0),
    num_bins=2,
    buffer_size=8,
    pick_window=4,
    close_policy=FullestOnConsecutiveRejectsPolicy(max_consecutive=4, min_fill_to_close=0.3),
    max_consecutive_rejects=50,
    enable_stability=False,
    allow_all_orientations=False,
)

boxes = generate_rajapack(n=100, seed=42)
strategy = get_strategy("baseline")
session = PackingSession(test_config)

print("Testing with 100 boxes...")
result = session.run(boxes, strategy)

print(f"\nResults:")
print(f"  Placed: {result.total_placed}, Rejected: {result.total_rejected}")
print(f"  Closed pallets: {len(result.closed_pallets)}")

if result.closed_pallets:
    print("\n✅ SUCCESS - Pallets closing!")
    for i, p in enumerate(result.closed_pallets):
        print(f"    Pallet {i+1}: {p.boxes_placed} boxes, {p.fill_rate:.1%} fill")
else:
    print("\n❌ No pallets closed")
    for i, p in enumerate(result.active_pallets):
        print(f"    Active {i+1}: {p.boxes_placed} boxes, {p.fill_rate:.1%} fill")
