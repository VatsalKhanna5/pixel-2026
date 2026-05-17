import sys; sys.path.insert(0, '.')
import numpy as np
from src.dataset.primitives import sample_layout
from src.dataset.em_simulation import simulate, FREQS
from src.dataset.physics_validator import _check_kk_causality, _check_passivity

rng = np.random.default_rng(0)
kk_failures = 0
pass_failures = 0
for i in range(50):
    layout, meta = sample_layout(rng)
    sim = simulate(layout, meta, substrate_id=0)
    pk, pd = _check_passivity(sim['s11_mag'], sim['s21_mag'])
    kk_ok, kd = _check_kk_causality(sim['s21_complex'])
    ptype = meta['type']
    if not kk_ok:
        kk_failures += 1
        print(f'  [{i}] type={ptype}  kk_residual={kd["kk_residual"]:.5f}')
    if not pk:
        pass_failures += 1
        print(f'  [{i}] type={ptype}  passivity_max={pd["passivity_max"]:.5f}')
print(f'KK failures: {kk_failures}/50   Passivity failures: {pass_failures}/50')
