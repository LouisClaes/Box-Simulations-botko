#!/usr/bin/env python3
"""
Test to validate pallet closing logic works correctly.
Forces rejections by using many boxes to fill up pallets.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import BinConfig
from dataset.generator import generate_rajapack
from strategies.base_strategy import get_strategy
from simulator.session import PackingSession, SessionConfig
from simulator.close_policy_custom import FullestOnConsecutiveRejectsPolicy

print("="*70)
print("PALLET CLOSING LOGIC VALIDATION TEST")
print("="*70)

# Test configuration
config = SessionConfig(
    bin_config=BinConfig(length=1200.0, width=800.0, height=2700.0, resolution=10.0),
    num_bins=2,
    buffer_size=8,
    pick_window=4,
    close_policy=FullestOnConsecutiveRejectsPolicy(
        max_consecutive=4,    # Close after 4 consecutive rejections
        min_fill_to_close=0.3  # Only close if >=30% full
    ),
    max_consecutive_rejects=200,  # Allow many rejects for testing
    enable_stability=False,
    allow_all_orientations=False,
)

# Test with increasing box counts to force rejections
test_cases = [
    (100, "Small dataset - may not trigger closures"),
    (200, "Medium dataset - likely some closures"),
    (300, "Large dataset - should trigger closures"),
]

for n_boxes, description in test_cases:
    print(f"\n{'─'*70}")
    print(f"TEST: {n_boxes} boxes - {description}")
    print(f"{'─'*70}")

    boxes = generate_rajapack(n=n_boxes, seed=42)
    strategy = get_strategy("baseline")
    session = PackingSession(config)

    print(f"Running simulation with {n_boxes} boxes...")
    result = session.run(boxes, strategy)

    print(f"\nRESULTS:")
    print(f"  Boxes placed:    {result.total_placed}")
    print(f"  Boxes rejected:  {result.total_rejected}")
    print(f"  Pallets closed:  {len(result.closed_pallets)}")
    print(f"  Active pallets:  {len(result.active_pallets)}")

    if result.closed_pallets:
        print(f"\n  ✅ CLOSED PALLETS (logic working!):")
        for i, pallet in enumerate(result.closed_pallets):
            print(f"    Pallet {i+1}: {pallet.boxes_placed} boxes, {pallet.fill_rate:.1%} fill, {pallet.max_height:.0f}mm height")
    else:
        print(f"\n  ❌ NO PALLETS CLOSED")

    if result.active_pallets:
        print(f"\n  Active pallets still open:")
        for i, pallet in enumerate(result.active_pallets):
            print(f"    Pallet {i+1}: {pallet.boxes_placed} boxes, {pallet.fill_rate:.1%} fill, {pallet.max_height:.0f}mm height")

print("\n" + "="*70)
print("VALIDATION SUMMARY")
print("="*70)

if any(len(result.closed_pallets) > 0 for n, _ in test_cases for result in [session.result()]):
    print("✅ SUCCESS: Pallet closing logic is working!")
    print("   Pallets closed when they filled up and couldn't accept more boxes.")
else:
    print("⚠️  WARNING: No pallets closed in any test case.")
    print("   This could mean:")
    print("   - Boxes fit too easily (all placed successfully)")
    print("   - Close threshold too high (min_fill_to_close)")
    print("   - Not enough consecutive rejections (max_consecutive)")

print("\nRecommendation: Check the printed output above for '[PALLET CLOSED]' messages.")
print("="*70)
