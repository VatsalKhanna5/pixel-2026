"""
src/dataset/primitives.py
PIXEL-2026 — RF Primitive Generators

Each generator returns a binary 15×15 numpy array (uint8) with conductor=1, void=0.
Port pixels are guaranteed to be conductor=1.

Primitives implemented:
  0. MicrostripLine        — straight horizontal line, P1→P2
  1. WidebandTaper         — tapered width line
  2. QuarterWaveShuntStub  — λ/4 shunt stub off centre
  3. HalfWaveResonator     — λ/2 shunt resonator
  4. NotchFilter           — narrow gap notch
  5. CoupledHalfWave       — edge-coupled λ/2 resonators
  6. InterdigitalBandpass  — interdigital fingers
  7. RingResonator         — square ring resonator
  8. SplitRingResonator    — split-ring resonator
  9. StubLoaded            — multi-stub loaded line
 10. CascadedResonators    — two stubs at different positions

All accept a `rng` (np.random.Generator) for reproducible perturbation.
All accept `sigma` = perturbation strength ∈ [0, 1].
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable

# Grid constants
H, W = 15, 15
PORT1 = (7, 0)    # (row, col) left-edge centre
PORT2 = (7, 14)   # (row, col) right-edge centre


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty() -> np.ndarray:
    return np.zeros((H, W), dtype=np.uint8)


def _set_ports(layout: np.ndarray) -> np.ndarray:
    """Ensure port pixels are always conductor."""
    layout[PORT1] = 1
    layout[PORT2] = 1
    return layout


def _clip(v, lo, hi):
    return int(np.clip(round(v), lo, hi))


def _draw_hline(layout, row, col_start, col_end):
    r = _clip(row, 0, H - 1)
    c0 = _clip(col_start, 0, W - 1)
    c1 = _clip(col_end, 0, W - 1)
    if c0 > c1:
        c0, c1 = c1, c0
    layout[r, c0:c1 + 1] = 1


def _draw_vline(layout, col, row_start, row_end):
    c = _clip(col, 0, W - 1)
    r0 = _clip(row_start, 0, H - 1)
    r1 = _clip(row_end, 0, H - 1)
    if r0 > r1:
        r0, r1 = r1, r0
    layout[r0:r1 + 1, c] = 1


def _draw_rect_ring(layout, r0, c0, r1, c1):
    """Draw a hollow rectangle (ring) border."""
    r0 = _clip(r0, 0, H - 1)
    r1 = _clip(r1, 0, H - 1)
    c0 = _clip(c0, 0, W - 1)
    c1 = _clip(c1, 0, W - 1)
    layout[r0, c0:c1 + 1] = 1   # top
    layout[r1, c0:c1 + 1] = 1   # bottom
    layout[r0:r1 + 1, c0] = 1   # left
    layout[r0:r1 + 1, c1] = 1   # right


# ---------------------------------------------------------------------------
# Primitive generators
# ---------------------------------------------------------------------------

def microstrip_line(rng: np.random.Generator, sigma: float = 0.15) -> tuple[np.ndarray, dict]:
    """Straight horizontal microstrip line, width 1–3 pixels."""
    layout = _empty()
    width = _clip(rng.normal(1.5, sigma * 2), 1, 3)
    centre_row = 7
    half = width // 2
    for drow in range(-half, half + 1):
        _draw_hline(layout, centre_row + drow, 0, W - 1)
    _set_ports(layout)
    return layout, {"type": 0, "width": width, "centre_row": centre_row}


def wideband_taper(rng: np.random.Generator, sigma: float = 0.15) -> tuple[np.ndarray, dict]:
    """Linearly tapered width from wide at port to narrow at centre and back."""
    layout = _empty()
    w_edge = _clip(rng.normal(3, sigma * 3), 2, 5)
    w_centre = _clip(rng.normal(1, sigma * 2), 1, 2)
    for col in range(W):
        # taper: wide at edges, narrow at centre
        t = abs(col - (W - 1) / 2) / ((W - 1) / 2)   # 0 at centre, 1 at edges
        w = max(1, round(w_centre + (w_edge - w_centre) * t))
        half = w // 2
        for drow in range(-half, half + 1):
            r = _clip(7 + drow, 0, H - 1)
            layout[r, col] = 1
    _set_ports(layout)
    return layout, {"type": 1, "w_edge": w_edge, "w_centre": w_centre}


def quarter_wave_shunt_stub(rng: np.random.Generator, sigma: float = 0.15) -> tuple[np.ndarray, dict]:
    """Through-line at row 7, single shunt stub perpendicular at random column."""
    layout = _empty()
    # Through-line
    _draw_hline(layout, 7, 0, W - 1)
    # Stub: connects from row 7 upward (or downward)
    stub_col = _clip(rng.normal(7, sigma * 5), 2, 12)
    stub_len = _clip(rng.normal(4, sigma * 3), 2, 7)
    direction = rng.choice([-1, 1])  # up or down
    stub_end_row = _clip(7 + direction * stub_len, 0, H - 1)
    _draw_vline(layout, stub_col, 7, stub_end_row)
    _set_ports(layout)
    return layout, {"type": 2, "stub_col": stub_col, "stub_len": stub_len, "direction": int(direction)}


def half_wave_resonator(rng: np.random.Generator, sigma: float = 0.15) -> tuple[np.ndarray, dict]:
    """Through-line + open λ/2 shunt resonator (longer stub)."""
    layout = _empty()
    _draw_hline(layout, 7, 0, W - 1)
    stub_col = _clip(rng.normal(7, sigma * 4), 2, 12)
    stub_len = _clip(rng.normal(6, sigma * 3), 4, H - 2)
    direction = rng.choice([-1, 1])
    stub_end_row = _clip(7 + direction * stub_len, 0, H - 1)
    _draw_vline(layout, stub_col, 7, stub_end_row)
    _set_ports(layout)
    return layout, {"type": 3, "stub_col": stub_col, "stub_len": stub_len}


def notch_filter(rng: np.random.Generator, sigma: float = 0.15) -> tuple[np.ndarray, dict]:
    """Through-line with a single-pixel gap at a notch position."""
    layout = _empty()
    _draw_hline(layout, 7, 0, W - 1)
    notch_col = _clip(rng.normal(7, sigma * 4), 2, 12)
    layout[7, notch_col] = 0  # break the line
    _set_ports(layout)
    return layout, {"type": 4, "notch_col": notch_col}


def coupled_half_wave(rng: np.random.Generator, sigma: float = 0.15) -> tuple[np.ndarray, dict]:
    """Two parallel horizontal lines (feed + coupled resonator)."""
    layout = _empty()
    gap = _clip(rng.normal(2, sigma * 2), 1, 4)
    res_start = _clip(rng.normal(3, sigma * 3), 1, 4)
    res_end   = _clip(rng.normal(11, sigma * 3), 10, 13)
    # Through-line at row 7
    _draw_hline(layout, 7, 0, W - 1)
    # Coupled resonator at row 7 - gap
    coupled_row = _clip(7 - gap, 0, H - 1)
    _draw_hline(layout, coupled_row, res_start, res_end)
    _set_ports(layout)
    return layout, {"type": 5, "gap": gap, "res_start": res_start, "res_end": res_end}


def interdigital_bandpass(rng: np.random.Generator, sigma: float = 0.15) -> tuple[np.ndarray, dict]:
    """Interdigital fingers: alternating up/down stubs from a central through-line."""
    layout = _empty()
    _draw_hline(layout, 7, 0, W - 1)
    n_fingers = _clip(rng.normal(4, sigma * 3), 2, 6)
    spacing = _clip((W - 2) / max(n_fingers, 1), 1, 5)
    finger_len = _clip(rng.normal(4, sigma * 3), 2, 6)
    for i in range(n_fingers):
        col = _clip(1 + i * spacing, 1, W - 2)
        direction = 1 if i % 2 == 0 else -1
        end_row = _clip(7 + direction * finger_len, 0, H - 1)
        _draw_vline(layout, col, 7, end_row)
    _set_ports(layout)
    return layout, {"type": 6, "n_fingers": n_fingers, "finger_len": finger_len}


def ring_resonator(rng: np.random.Generator, sigma: float = 0.15) -> tuple[np.ndarray, dict]:
    """Rectangular ring resonator with a through-line feed."""
    layout = _empty()
    _draw_hline(layout, 7, 0, W - 1)
    # Ring centred at column 7
    ring_size = _clip(rng.normal(5, sigma * 3), 3, 7)
    half = ring_size // 2
    cr, cc = 5, 7   # ring centre (slightly above through-line)
    cr = _clip(rng.normal(5, sigma * 2), 2, 9)
    _draw_rect_ring(layout, cr - half, cc - half, cr + half, cc + half)
    # Coupling gap: break ring bottom at through-line junction
    layout[_clip(cr + half, 0, H - 1), cc] = 0
    _set_ports(layout)
    return layout, {"type": 7, "ring_size": ring_size, "cr": int(cr), "cc": cc}


def split_ring_resonator(rng: np.random.Generator, sigma: float = 0.15) -> tuple[np.ndarray, dict]:
    """SRR: ring with a split gap + through-line feed."""
    layout = _empty()
    _draw_hline(layout, 7, 0, W - 1)
    ring_size = _clip(rng.normal(5, sigma * 3), 3, 7)
    half = ring_size // 2
    cr = _clip(rng.normal(4, sigma * 2), 2, 8)
    cc = _clip(rng.normal(7, sigma * 2), 4, 10)
    _draw_rect_ring(layout, cr - half, cc - half, cr + half, cc + half)
    # Two gaps on the same side (split)
    gap_col = _clip(cc + rng.integers(-1, 2), cc - half + 1, cc + half - 1)
    layout[cr - half, gap_col] = 0
    layout[cr - half, _clip(gap_col + 1, 0, W - 1)] = 0
    _set_ports(layout)
    return layout, {"type": 8, "ring_size": ring_size, "cr": int(cr), "cc": int(cc)}


def stub_loaded_line(rng: np.random.Generator, sigma: float = 0.15) -> tuple[np.ndarray, dict]:
    """Through-line with 2 shunt stubs at different positions."""
    layout = _empty()
    _draw_hline(layout, 7, 0, W - 1)
    positions = sorted(rng.choice(range(2, 13), size=2, replace=False))
    lengths = [_clip(rng.normal(3, sigma * 3), 1, 6) for _ in range(2)]
    directions = [rng.choice([-1, 1]) for _ in range(2)]
    for col, length, direction in zip(positions, lengths, directions):
        end_row = _clip(7 + direction * length, 0, H - 1)
        _draw_vline(layout, col, 7, end_row)
    _set_ports(layout)
    return layout, {"type": 9, "positions": [int(p) for p in positions], "lengths": [int(l) for l in lengths]}


def cascaded_resonators(rng: np.random.Generator, sigma: float = 0.15) -> tuple[np.ndarray, dict]:
    """Two resonator stubs at distinct positions with different lengths."""
    layout = _empty()
    _draw_hline(layout, 7, 0, W - 1)
    col1 = _clip(rng.normal(4, sigma * 2), 2, 6)
    col2 = _clip(rng.normal(10, sigma * 2), 8, 12)
    len1  = _clip(rng.normal(5, sigma * 3), 3, 7)
    len2  = _clip(rng.normal(3, sigma * 3), 1, 6)
    dir1  = 1   # upward
    dir2  = -1  # downward
    _draw_vline(layout, col1, 7, _clip(7 - len1, 0, H - 1))
    _draw_vline(layout, col2, 7, _clip(7 + len2, 0, H - 1))
    _set_ports(layout)
    return layout, {"type": 10, "col1": col1, "col2": col2, "len1": len1, "len2": len2}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PRIMITIVE_GENERATORS: list[Callable] = [
    microstrip_line,
    wideband_taper,
    quarter_wave_shunt_stub,
    half_wave_resonator,
    notch_filter,
    coupled_half_wave,
    interdigital_bandpass,
    ring_resonator,
    split_ring_resonator,
    stub_loaded_line,
    cascaded_resonators,
]

PRIMITIVE_NAMES = [
    "MicrostripLine",
    "WidebandTaper",
    "QuarterWaveShuntStub",
    "HalfWaveResonator",
    "NotchFilter",
    "CoupledHalfWave",
    "InterdigitalBandpass",
    "RingResonator",
    "SplitRingResonator",
    "StubLoadedLine",
    "CascadedResonators",
]

N_PRIMITIVES = len(PRIMITIVE_GENERATORS)

# Target proportion for balanced generation (sum ≈ 1.0)
PRIMITIVE_WEIGHTS = np.array([
    0.10,  # microstrip_line
    0.07,  # wideband_taper
    0.10,  # quarter_wave_shunt_stub
    0.08,  # half_wave_resonator
    0.07,  # notch_filter
    0.10,  # coupled_half_wave
    0.10,  # interdigital_bandpass
    0.10,  # ring_resonator
    0.10,  # split_ring_resonator
    0.09,  # stub_loaded_line
    0.09,  # cascaded_resonators
], dtype=np.float32)
PRIMITIVE_WEIGHTS /= PRIMITIVE_WEIGHTS.sum()


def local_mutate(layout: np.ndarray, rng: np.random.Generator, n_flips: int | None = None) -> np.ndarray:
    """Randomly flip n_flips non-port pixels."""
    if n_flips is None:
        n_flips = int(rng.integers(1, 6))
    out = layout.copy()
    # Exclude port pixels
    candidates = [(r, c) for r in range(H) for c in range(W)
                  if (r, c) != PORT1 and (r, c) != PORT2]
    chosen = rng.choice(len(candidates), size=min(n_flips, len(candidates)), replace=False)
    for idx in chosen:
        r, c = candidates[idx]
        out[r, c] = 1 - out[r, c]
    return out


def combine_layouts(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Logical OR of two layouts (union of conductors), preserving ports."""
    out = np.clip(a.astype(np.int16) + b.astype(np.int16), 0, 1).astype(np.uint8)
    out[PORT1] = 1
    out[PORT2] = 1
    return out


def sample_layout(rng: np.random.Generator, sigma: float = 0.15) -> tuple[np.ndarray, dict]:
    """
    Sample one layout by:
    1. Picking a primitive type proportional to PRIMITIVE_WEIGHTS
    2. Calling its generator with random perturbation
    3. With prob 0.3: apply local_mutate
    4. With prob 0.2: combine with another randomly sampled primitive
    """
    idx = rng.choice(N_PRIMITIVES, p=PRIMITIVE_WEIGHTS)
    layout, meta = PRIMITIVE_GENERATORS[idx](rng, sigma)

    if rng.random() < 0.30:
        n_flips = int(rng.integers(1, 6))
        layout = local_mutate(layout, rng, n_flips)

    if rng.random() < 0.20:
        idx2 = rng.choice(N_PRIMITIVES, p=PRIMITIVE_WEIGHTS)
        layout2, _ = PRIMITIVE_GENERATORS[idx2](rng, sigma)
        layout = combine_layouts(layout, layout2)
        meta["combined_with"] = int(idx2)

    # Final port guarantee
    layout[PORT1] = 1
    layout[PORT2] = 1

    return layout, meta
