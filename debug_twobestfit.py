import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Box, BinConfig
from simulator.session import PackingSession, SessionConfig
from simulator.close_policy import HeightClosePolicy
from strategies.base_strategy import get_multibin_strategy
from dataset.generator import generate_rajapack

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

strategy = get_multibin_strategy("two_bounded_best_fit")
session = PackingSession(BOTKO_SESSION_CONFIG)
boxes = generate_rajapack(50, seed=42)

def on_step(step_num, step_result, obs):
    if step_result.placed:
        pass
    else:
        print(f"Step {step_num}: Box {step_result.box.id if step_result.box else 'None'} REJECTED.")

print("Starting debug run for two_bounded_best_fit...")
res = session.run(boxes, strategy, on_step=on_step)
print(f"Placed: {res.total_placed}, Rejected: {res.total_rejected}")
for i, st in enumerate(session.stations):
    print(f"Pallet {i} fill: {st.bin_state.get_fill_rate():.2%}")
