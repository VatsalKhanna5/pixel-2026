import sys; sys.path.insert(0, '.')
import numpy as np
from src.dataset.primitives import sample_layout
from src.dataset.em_simulation import simulate, FREQS
from src.dataset.connectivity import is_connected

rng = np.random.default_rng(0)
bad = []
for i in range(500):
    layout, meta = sample_layout(rng)
    if not is_connected(layout):
        continue
    sim = simulate(layout, meta, substrate_id=0)
    diffs = np.abs(np.diff(sim['s21_mag']))
    max_jump = float(diffs.max())
    if max_jump > 0.30:
        bad.append((i, meta['type'], max_jump, diffs.argmax()))
        
print(f"Bad structures: {len(bad)}")
for rec in bad[:20]:
    print(f"  idx={rec[0]} type={rec[1]} max_jump={rec[2]:.4f} at freq_idx={rec[3]}")
