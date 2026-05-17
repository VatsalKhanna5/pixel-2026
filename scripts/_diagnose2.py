import sys; sys.path.insert(0, '.')
import numpy as np
from src.dataset.primitives import sample_layout
from src.dataset.em_simulation import simulate, FREQS
from src.dataset.physics_validator import _check_spectral_smoothness, _check_passivity

rng = np.random.default_rng(0)
smooth_failures = 0
for i in range(50):
    layout, meta = sample_layout(rng)
    sim = simulate(layout, meta, substrate_id=0)
    sm_ok, sd = _check_spectral_smoothness(sim['s21_mag'])
    if not sm_ok:
        smooth_failures += 1
        print(f'  [{i}] type={meta["type"]}  max_jump={sd["spectral_max_jump"]:.5f}')
print(f'Smooth failures: {smooth_failures}/50')
