#!/usr/bin/env python3
"""
Test slow strategies to determine if they're viable for demo run.
Tests with 50 boxes (smaller than 400) to get quick timing estimate.
"""
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset.generator import generate_rajapack
from strategies.base_strategy import get_strategy
from simulator.session import PackingSession
from run_overnight_botko_telegram import BOTKO_SESSION_CONFIG

print("=" * 70)
print("SLOW STRATEGY VALIDATION TEST")
print("=" * 70)
print("Testing: lookahead, hybrid_adaptive")
print("Dataset: 50 Rajapack boxes (reduced from 400)")
print("Timeout: 5 minutes per strategy")
print("=" * 70)

SLOW_STRATEGIES = ["lookahead", "hybrid_adaptive"]
boxes = generate_rajapack(n=50, seed=42)

results = {}

for strategy_name in SLOW_STRATEGIES:
    print(f"\n[TEST] {strategy_name}")
    print(f"  Boxes: 50")

    try:
        strategy = get_strategy(strategy_name)
        session = PackingSession(BOTKO_SESSION_CONFIG)

        start = time.perf_counter()
        result = session.run(boxes, strategy)
        elapsed = time.perf_counter() - start

        print(f"  ✓ Completed in {elapsed:.1f}s ({elapsed/60:.2f} min)")
        print(f"    Placed: {result.total_placed}/{result.total_boxes}")
        print(f"    Closed pallets: {result.pallets_closed}")
        print(f"    Avg fill: {result.avg_closed_fill:.1%}")

        # Extrapolate to 400 boxes
        time_per_box = elapsed / 50
        estimated_400 = time_per_box * 400
        print(f"    Estimated time for 400 boxes: {estimated_400:.1f}s ({estimated_400/60:.1f} min)")

        results[strategy_name] = {
            "status": "✓ SUCCESS",
            "time_50_boxes": elapsed,
            "time_per_box": time_per_box,
            "estimated_400_boxes": estimated_400,
            "viable": estimated_400 < 600,  # < 10 minutes for 400 boxes
        }

    except KeyboardInterrupt:
        print(f"  ✗ Interrupted by user")
        results[strategy_name] = {
            "status": "✗ INTERRUPTED",
            "viable": False,
        }
        break

    except Exception as e:
        print(f"  ✗ Error: {str(e)[:100]}")
        results[strategy_name] = {
            "status": f"✗ ERROR: {str(e)[:50]}",
            "viable": False,
        }

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

for strategy, data in results.items():
    print(f"\n{strategy}:")
    print(f"  Status: {data['status']}")
    if data.get('time_50_boxes'):
        print(f"  Time (50 boxes): {data['time_50_boxes']:.1f}s")
        print(f"  Estimated (400 boxes): {data['estimated_400_boxes']:.1f}s ({data['estimated_400_boxes']/60:.1f} min)")
        print(f"  Viable for demo: {'✓ YES' if data['viable'] else '✗ NO (too slow)'}")

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

viable_count = sum(1 for d in results.values() if d.get('viable'))
total_count = len(results)

if viable_count == total_count:
    print("✓ All slow strategies are VIABLE for demo run")
    print("  Proceed with full strategy list (21 fast + 2 slow)")
elif viable_count > 0:
    print(f"⚠️  {viable_count}/{total_count} slow strategies are viable")
    print("  Consider running demo with viable strategies only")
    viable_strats = [s for s, d in results.items() if d.get('viable')]
    print(f"  Viable: {viable_strats}")
else:
    print("✗ NO slow strategies are viable for demo run")
    print("  Exclude ALL slow strategies from demo")
    print("  Set EXCLUDED_STRATEGIES = ['lookahead', 'hybrid_adaptive', 'selective_hyper_heuristic']")

print("=" * 70)
