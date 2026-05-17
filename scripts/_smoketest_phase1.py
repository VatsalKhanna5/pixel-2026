import sys; sys.path.insert(0, '.')

print('Testing primitives ...')
from src.dataset.primitives import sample_layout, PRIMITIVE_NAMES, PRIMITIVE_GENERATORS
import numpy as np
rng = np.random.default_rng(42)
for gen, name in zip(PRIMITIVE_GENERATORS, PRIMITIVE_NAMES):
    layout, meta = gen(rng, sigma=0.15)
    assert layout.shape == (15, 15), f'shape error: {name}'
    assert layout.dtype == np.uint8, f'dtype error: {name}'
    assert layout[7, 0] == 1 and layout[7, 14] == 1, f'port error: {name}'
print('  All 11 primitives: OK')

print('Testing connectivity ...')
from src.dataset.connectivity import is_connected, check_drc
layout2, _ = sample_layout(rng)
conn = is_connected(layout2)
print(f'  sample_layout connected: {conn}')

print('Testing em_simulation ...')
from src.dataset.em_simulation import simulate, FREQS
layout3, meta3 = sample_layout(rng)
sim = simulate(layout3, meta3, substrate_id=0)
assert sim['s11_mag'].shape == (100,)
assert sim['s21_mag'].shape == (100,)
power_max = float((sim['s11_mag']**2 + sim['s21_mag']**2).max())
assert power_max <= 1.01, f'passivity failed: max={power_max:.4f}'
print(f'  sim OK  passivity_ok={sim["passivity_ok"]}  kk_residual={sim["kk_residual"]:.4f}  power_max={power_max:.4f}')

print('Testing physics_validator ...')
from src.dataset.physics_validator import validate_record, check_dataset_quality_gates
report = validate_record(
    layout3, sim['s11_mag'], sim['s21_mag'],
    sim['s11_phase'], sim['s21_phase'], sim['s21_complex']
)
print(f'  validity_flag={report.validity_flag}  passivity={report.passivity_ok}  kk={report.kk_ok}  fail={report.fail_reasons}')

print('Testing hdf5_writer ...')
from src.dataset.hdf5_writer import CheckpointWriter, resume_from_checkpoint
import tempfile, os
tmp = tempfile.mktemp(suffix='.h5')
try:
    rec = {
        'layout': layout3,
        's11_mag': sim['s11_mag'], 's21_mag': sim['s21_mag'],
        's11_phase': sim['s11_phase'], 's21_phase': sim['s21_phase'],
        'substrate_id': 0,
        'resonance_freqs': sim['resonance_freqs'],
        'q_factors': sim['q_factors'],
        'validity_flag': True,
        'primitive_type': 0,
    }
    with CheckpointWriter(tmp, batch_size=2) as w:
        for _ in range(5):
            w.write_record(rec)
    nxt = resume_from_checkpoint(tmp)
    print(f'  HDF5 write 5 records OK  next_id={nxt}')
finally:
    for ext in ['', '.checkpoint.json', '.lock']:
        p = tmp if ext == '' else tmp.replace('.h5', ext)
        if os.path.exists(p):
            os.unlink(p)

print()
print('ALL SMOKE TESTS PASSED')
