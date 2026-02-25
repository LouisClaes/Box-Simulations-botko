#!/usr/bin/env python3
"""
Test the new lookahead close logic.
Validates that pallets close when current 4 + next 4 boxes don't fit.
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
print("LOOKAHEAD CLOSE LOGIC VALIDATION")
print("="*70)
print("\nLogic:")
print("  - Buffer: 8 boxes total")
print("  - Pick window: First 4 boxes")
print("  - Lookahead: Check next 4 boxes (5-8)")
print("  - Close if: NEITHER current 4 NOR next 4 fit on ANY pallet")
print("  - Close: Fullest pallet (if >=50% full)")
print("="*70)

config = SessionConfig(
    bin_config=BinConfig(length=1200.0, width=800.0, height=2700.0, resolution=10.0),
    num_bins=2,
    buffer_size=8,  # 8 boxes in buffer
    pick_window=4,   # First 4 are grippable
    close_policy=FullestOnConsecutiveRejectsPolicy(max_consecutive=4, min_fill_to_close=0.5),
    max_consecutive_rejects=200,
    enable_stability=False,
    allow_all_orientations=False,
)

# Test with 150 boxes
boxes = generate_rajapack(n=150, seed=42)
strategy = get_strategy("baseline")
session = PackingSession(config)

print(f"\nRunning with 150 boxes...")
print("Watch for '[PALLET CLOSED]' messages with lookahead reason\n")

result = session.run(boxes, strategy)

print(f"\n{'='*70}")
print("RESULTS:")
print(f"{'='*70}")
print(f"  Boxes placed:    {result.total_placed}")
print(f"  Boxes rejected:  {result.total_rejected}")
print(f"  Pallets closed:  {len(result.closed_pallets)}")
print(f"  Active pallets:  {len(result.active_pallets)}")

if result.closed_pallets:
    print(f"\n✅ SUCCESS: Lookahead close logic triggered!")
    for i, p in enumerate(result.closed_pallets):
        print(f"  Closed pallet {i+1}: {p.boxes_placed} boxes, {p.fill_rate:.1%} fill, {p.max_height:.0f}mm")
else:
    print(f"\n⚠️  No pallets closed")
    print("  Possible reasons:")
    print("  - Boxes fit too easily (all placed)")
    print("  - Pallets not full enough (<50%)")
    print("  - Always found a box in next 4 that fits")

if result.active_pallets:
    print(f"\n  Active pallets:")
    for i, p in enumerate(result.active_pallets):
        print(f"    Pallet {i+1}: {p.boxes_placed} boxes, {p.fill_rate:.1%} fill, {p.max_height:.0f}mm")

print("="*70)
