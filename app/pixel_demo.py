"""
app/pixel_demo.py  ·  PIXEL-2026 Streamlit Demo  ·  v2 (complete rewrite)
==========================================================================
Interactive demo: specify target S-parameters → generate RF layout →
surrogate prediction → OpenEMS ground-truth verification → Gerber export.

Run from project root:
    bash scripts/launch_streamlit.sh          # auto-detects GPU, sets tunnel instructions

Remote access from local PC (two-step):
    1. SSH tunnel:  ssh -L 8501:localhost:8501 ec_23104075@10.10.11.201
    2. Browser:     http://localhost:8501
"""

from __future__ import annotations

import io
import json
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn.functional as F
from plotly.subplots import make_subplots

# ── Project path ─────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from src.dataset.connectivity import is_connected, PORT1, PORT2
from src.guidance.cfg import discrete_cfg
from src.guidance.physics_guidance import guided_reverse_step
from src.models.connectivity_disc import ConnectivityDiscriminator
from src.models.denoiser import PixelDenoiser
from src.models.diffusion import D3PMAbsorbing
from src.models.spectral_encoder import SpectralEncoder
from src.models.surrogate import PhysicsSurrogate, SurrogateEnsemble

# ── Constants ─────────────────────────────────────────────────────────────────
FREQS_GHZ = np.linspace(0.5, 20.0, 100, dtype=np.float32)
FREQS_HZ  = FREQS_GHZ * 1e9
N_FREQ    = 100
H, W      = 15, 15
PIXEL_MM  = 0.5   # physical pixel pitch

PORT_MAP = np.zeros((H, W), dtype=np.float32)
PORT_MAP[PORT1[0], PORT1[1]] = 1.0
PORT_MAP[PORT2[0], PORT2[1]] = 1.0

SURR_PATHS = [str(_ROOT / f"experiments/surrogate_v1/surrogate_k{k}_best.pt")
              for k in range(5)]
DENOISER_PATH = str(_ROOT / "experiments/denoiser_v1/denoiser_best.pt")
DISC_PATH     = str(_ROOT / "experiments/discriminator_v1/disc_best.pt")

SUBSTRATES = {
    "Rogers 4003C": {"id": 0, "eps_r": 3.55, "tan_d": 0.0027, "note": "High-freq standard"},
    "FR4":          {"id": 1, "eps_r": 4.40, "tan_d": 0.0200, "note": "Low-cost PCB"},
    "Rogers 5880":  {"id": 2, "eps_r": 2.20, "tan_d": 0.0009, "note": "PTFE / mmWave"},
    "Alumina":      {"id": 3, "eps_r": 9.80, "tan_d": 0.0001, "note": "MMIC / ceramic"},
}

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="PIXEL · EM Synthesizer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ════════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ════════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading PIXEL models…")
def load_models() -> tuple[dict | None, str | None]:
    """Load all trained model checkpoints once, cache across reruns."""
    missing = [p for p in SURR_PATHS + [DENOISER_PATH, DISC_PATH]
               if not Path(p).exists()]
    if missing:
        return None, "Missing checkpoints:\n" + "\n".join(missing)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Surrogate ensemble (K=5)
        ens = SurrogateEnsemble.load(SURR_PATHS, device=str(device))
        ens.eval()

        # Denoiser + encoder
        ck = torch.load(DENOISER_PATH, map_location=device, weights_only=False)
        denoiser = PixelDenoiser()
        encoder  = SpectralEncoder()
        state_key = "ema_state" if "ema_state" in ck else "denoiser_state"
        denoiser.load_state_dict(ck[state_key])
        encoder.load_state_dict(ck["encoder_state"])
        denoiser.eval().to(device)
        encoder.eval().to(device)

        # Connectivity discriminator
        ck_d = torch.load(DISC_PATH, map_location=device, weights_only=False)
        disc = ConnectivityDiscriminator()
        disc.load_state_dict(ck_d["model_state"])
        disc.eval().to(device)

        # D3PM — must call .to(device) so internal schedule tensors are on same device
        # as x_t / t_tensor; otherwise _get() raises "indices on wrong device"
        diffusion = D3PMAbsorbing(T=1000)
        diffusion.to(device)

        pm = torch.from_numpy(PORT_MAP).unsqueeze(0).unsqueeze(0).to(device)

        return {
            "surrogate": ens, "denoiser": denoiser,
            "encoder": encoder, "discriminator": disc,
            "diffusion": diffusion, "port_map": pm, "device": device,
        }, None

    except Exception:
        return None, traceback.format_exc()


# ════════════════════════════════════════════════════════════════════════════════
# SPECIFICATION BUILDERS
# ════════════════════════════════════════════════════════════════════════════════

def _pack(s11_mag, s21_mag, s11_ph, s21_ph) -> np.ndarray:
    wrap = lambda p: ((p + np.pi) % (2 * np.pi) - np.pi)
    return np.stack([
        s11_mag.astype(np.float32), s21_mag.astype(np.float32),
        (wrap(s11_ph) / np.pi).astype(np.float32),
        (wrap(s21_ph) / np.pi).astype(np.float32),
    ]).astype(np.float32)   # (4, 100)


def spec_bandpass(fc_ghz: float, bw_ghz: float) -> np.ndarray:
    fc, bw  = fc_ghz * 1e9, bw_ghz * 1e9
    s21     = np.exp(-np.log(2) * ((FREQS_HZ - fc) / (bw / 2)) ** 2)
    s21     = np.clip(s21, 1e-4, 1.0)
    s11     = np.sqrt(np.clip(1 - s21 ** 2, 1e-4, 1.0))
    gd      = 0.3e-9
    phi21   = -2 * np.pi * FREQS_HZ * gd
    return _pack(s11, s21, phi21 + np.pi, phi21)


def spec_bandstop(fc_ghz: float, bw_ghz: float, rej_db: float) -> np.ndarray:
    fc, bw  = fc_ghz * 1e9, bw_ghz * 1e9
    floor   = 10 ** (-abs(rej_db) / 20)
    s21     = 1.0 - (1.0 - floor) * np.exp(-np.log(2) * ((FREQS_HZ - fc) / (bw / 2)) ** 2)
    s21     = np.clip(s21, floor, 1.0)
    s11     = np.sqrt(np.clip(1 - s21 ** 2, 1e-4, 1.0))
    phi21   = -2 * np.pi * FREQS_HZ * 0.3e-9
    return _pack(s11, s21, phi21 + np.pi, phi21)


def spec_lowpass(fc_ghz: float, order: int = 3) -> np.ndarray:
    fc   = fc_ghz * 1e9
    s21  = 1.0 / np.sqrt(1.0 + (FREQS_HZ / fc) ** (2 * order))
    s21  = np.clip(s21, 1e-4, 1.0)
    s11  = np.sqrt(np.clip(1 - s21 ** 2, 1e-4, 1.0))
    phi21 = -2 * np.pi * FREQS_HZ * 0.3e-9
    return _pack(s11, s21, phi21 + np.pi, phi21)


def spec_wideband() -> np.ndarray:
    s21 = np.ones(N_FREQ, dtype=np.float32) * 0.92
    s11 = np.ones(N_FREQ, dtype=np.float32) * 0.18
    phi21 = -2 * np.pi * FREQS_HZ * 0.2e-9
    return _pack(s11, s21, phi21 + np.pi, phi21)


# ════════════════════════════════════════════════════════════════════════════════
# LAYOUT UTILITIES
# ════════════════════════════════════════════════════════════════════════════════

def remove_islands(layout: np.ndarray) -> np.ndarray:
    """BFS: zero out conductor pixels not reachable from either port."""
    from collections import deque
    lay = layout.copy()
    reachable: set = set()
    for start in [PORT1, PORT2]:
        if lay[start] == 0:
            continue
        q = deque([start])
        seen = {start}
        while q:
            r, c = q.popleft()
            reachable.add((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and lay[nr, nc] == 1 \
                        and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    q.append((nr, nc))
    for r in range(H):
        for c in range(W):
            if lay[r, c] == 1 and (r, c) not in reachable:
                lay[r, c] = 0
    return lay


def predict_surrogate(layout: np.ndarray, models: dict) -> dict:
    """Run surrogate ensemble on one layout; return predicted S-params."""
    device = models["device"]
    x_in   = torch.from_numpy(
        np.stack([layout.astype(np.float32), PORT_MAP], axis=0)[None]
    ).to(device)  # (1, 2, 15, 15)
    with torch.no_grad():
        mean, var = models["surrogate"](x_in)
    m = mean[0].cpu().numpy()   # (4, 100)
    return {
        "s11_mag": m[0], "s21_mag": m[1],
        "s11_ph":  m[2] * np.pi, "s21_ph": m[3] * np.pi,
        "sigma":   float(var[0].mean().sqrt().cpu()),
    }


def em_mse(target: np.ndarray, s21: np.ndarray, s11: np.ndarray) -> dict:
    mse_s21   = float(np.mean((s21 - target[1]) ** 2))
    mse_s11   = float(np.mean((s11 - target[0]) ** 2))
    joint     = (mse_s21 + mse_s11) / 2
    return {"S21 MSE": mse_s21, "S11 MSE": mse_s11,
            "Joint MSE": joint,
            "Coverage @0.001": joint < 0.001,
            "Coverage @0.010": joint < 0.010}


# ════════════════════════════════════════════════════════════════════════════════
# GENERATION
# ════════════════════════════════════════════════════════════════════════════════

def generate(
    y_star_np: np.ndarray,
    models: dict,
    *,
    T: int = 1000,
    alpha_max: float = 0.10,
    cfg_w: float = 2.0,
    use_guidance: bool = True,
    seed: int = 42,
    progress_bar=None,
) -> np.ndarray:
    """
    Run the full PIXEL reverse diffusion chain.
    Returns a cleaned binary (15,15) layout.
    """
    torch.manual_seed(seed)
    device    = models["device"]
    denoiser  = models["denoiser"]
    encoder   = models["encoder"]
    diffusion = models["diffusion"]
    surr      = models["surrogate"]
    disc      = models["discriminator"]
    pm        = models["port_map"]

    y_t = torch.from_numpy(y_star_np)[None].to(device)    # (1, 4, 100)
    with torch.no_grad():
        c_y = encoder(y_t)                                  # (1, 256)

    x = torch.full((1, H, W), diffusion.MASK, dtype=torch.long, device=device)
    thresh = int(T * 0.4) if use_guidance else 0

    for idx, t_val in enumerate(range(T, 0, -1)):
        t_vec = torch.full((1,), t_val, dtype=torch.long, device=device)
        x = guided_reverse_step(
            x, t_val, t_vec,
            denoiser, encoder, diffusion, surr, disc,
            pm, c_y, y_t,
            T=T, t_thresh=thresh,
            alpha_max=alpha_max if use_guidance else 0.0,
            epsilon=0.01, lambda_topo=1.0, lambda_mfg=0.5,
            g_max=1.0, cfg_w=cfg_w,
        )
        if progress_bar is not None and idx % 50 == 0:
            progress_bar.progress((idx + 1) / T)

    if progress_bar is not None:
        progress_bar.progress(1.0)

    return remove_islands(x[0].cpu().numpy().astype(np.uint8))


# ════════════════════════════════════════════════════════════════════════════════
# OPENEMS
# ════════════════════════════════════════════════════════════════════════════════

def run_em(layout: np.ndarray, substrate_id: int) -> dict | None:
    try:
        from src.dataset.openems_wrapper import simulate
        return simulate(layout, substrate_id=substrate_id)
    except Exception as e:
        st.error(f"OpenEMS failed: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════════
# GERBER EXPORT
# ════════════════════════════════════════════════════════════════════════════════

def _gbr_copper(layout: np.ndarray) -> str:
    """RS-274X copper top layer."""
    scale = 1_000_000  # mm → 4.6 integer units
    hdr = [
        "G04 PIXEL-2026 Layout — Copper Top*",
        "%FSLAX46Y46*%",
        "%MOMM*%",
        "%LPD*%",
        f"%ADD10R,{PIXEL_MM:.4f}X{PIXEL_MM:.4f}*%",
        "D10*",
        "G01*",
    ]
    flashes = []
    for r in range(H):
        for c in range(W):
            if layout[r, c] == 1:
                x = int(round((c + 0.5) * PIXEL_MM * scale))
                y = int(round((H - r - 0.5) * PIXEL_MM * scale))
                flashes.append(f"X{x:010d}Y{y:010d}D03*")
    return "\n".join(hdr + flashes + ["M02*"])


def _gbr_outline() -> str:
    """RS-274X board outline."""
    scale  = 1_000_000
    bx     = int(W * PIXEL_MM * scale)
    by     = int(H * PIXEL_MM * scale)
    hdr    = [
        "G04 PIXEL-2026 Layout — Board Outline*",
        "%FSLAX46Y46*%",
        "%MOMM*%",
        "%LPD*%",
        "%ADD11C,0.050000*%",
        "D11*", "G01*",
    ]
    rect   = [
        "X0000000000Y0000000000D02*",
        f"X{bx:010d}Y0000000000D01*",
        f"X{bx:010d}Y{by:010d}D01*",
        f"X0000000000Y{by:010d}D01*",
        "X0000000000Y0000000000D01*",
    ]
    return "\n".join(hdr + rect + ["M02*"])


def _gbr_soldermask() -> str:
    """RS-274X soldermask — expose all copper (no mask openings defined)."""
    return "\n".join([
        "G04 PIXEL-2026 Layout — Soldermask Top (clear all)*",
        "%FSLAX46Y46*%",
        "%MOMM*%",
        "%LPC*%",               # clear polarity — removes soldermask
        f"%ADD12R,{W*PIXEL_MM:.4f}X{H*PIXEL_MM:.4f}*%",
        "D12*",
        f"X{int(W*PIXEL_MM/2*1e6):010d}Y{int(H*PIXEL_MM/2*1e6):010d}D03*",
        "M02*",
    ])


def _excellon_drill() -> str:
    """Excellon drill: 0.8 mm connector holes at port centres."""
    # Port 1: row=7, col=0   Port 2: row=7, col=14
    def coord(r, c):
        x = (c + 0.5) * PIXEL_MM
        y = (H - r - 0.5) * PIXEL_MM
        return f"X{x:.3f}Y{y:.3f}"
    return "\n".join([
        "M48",
        "; PIXEL-2026 Drill File",
        "METRIC,LZ",
        "T01C0.800",     # 0.8 mm for SMA / edge-launch connector
        "%",
        coord(*PORT1),
        coord(*PORT2),
        "M30",
    ])


def _csv_layout(layout: np.ndarray) -> str:
    return "\n".join(",".join(str(v) for v in row) for row in layout)


def make_gerber_zip(
    layout: np.ndarray,
    spec_label: str,
    substrate: str,
    method: str,
    surr_metrics: dict,
    em_metrics: dict | None,
) -> bytes:
    """Package all Gerber files + metadata into a ZIP archive."""
    readme = f"""PIXEL-2026 Generated RF Layout — Gerber Package
================================================

Specification : {spec_label}
Substrate     : {substrate}
Generator     : {method}
Grid          : 15 × 15 pixels  ·  {PIXEL_MM} mm/pixel  ·  7.5 mm × 7.5 mm board
Port 1        : row 7, col 0   →  x = 0.25 mm, y = 3.75 mm
Port 2        : row 7, col 14  →  x = 7.25 mm, y = 3.75 mm

Performance:
  Surrogate Joint MSE  : {surr_metrics.get('Joint MSE', 'N/A'):.5f}
  EM Joint MSE         : {em_metrics.get('Joint MSE', 'N/A') if em_metrics else 'Not verified'}
  EM Coverage @0.001   : {em_metrics.get('Coverage @0.001', 'N/A') if em_metrics else 'Not verified'}

Files:
  copper_top.gbr     Top copper layer (RS-274X)
  board_outline.gbr  Board edge cuts (RS-274X)
  soldermask_top.gbr Soldermask top (RS-274X, clear-all)
  drill.drl          Connector holes, T01=0.8mm (Excellon)
  layout.csv         Binary pixel map  (15×15, 1=conductor)
  metrics.json       Generation metadata & EM results

Generated by PIXEL-2026 (AAAI-2027 submission candidate)
"""
    meta = json.dumps({
        "spec": spec_label, "substrate": substrate, "method": method,
        "layout_pixels": layout.tolist(),
        "surrogate_metrics": surr_metrics,
        "em_metrics": em_metrics,
        "physical": {"pixel_mm": PIXEL_MM, "board_mm": [W * PIXEL_MM, H * PIXEL_MM]},
    }, indent=2)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("copper_top.gbr",     _gbr_copper(layout))
        zf.writestr("board_outline.gbr",  _gbr_outline())
        zf.writestr("soldermask_top.gbr", _gbr_soldermask())
        zf.writestr("drill.drl",          _excellon_drill())
        zf.writestr("layout.csv",         _csv_layout(layout))
        zf.writestr("metrics.json",       meta)
        zf.writestr("README.txt",         readme)
    return buf.getvalue()


# ════════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ════════════════════════════════════════════════════════════════════════════════

_COLORS = {
    "target":    "#e74c3c",   # red-ish
    "surrogate": "#2980b9",   # blue
    "em":        "#27ae60",   # green
}


def fig_layout(layout: np.ndarray, title: str = "Generated Layout",
               size: int = 320) -> go.Figure:
    """Render 15×15 binary layout as a copper-coloured heatmap."""
    # Build RGBA image: conductor = copper gold, dielectric = light grey
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[layout == 0] = [240, 240, 240]   # dielectric: light grey
    img[layout == 1] = [184, 115, 51]    # conductor:  copper

    fig = go.Figure()
    fig.add_trace(go.Image(z=img))

    # Port markers
    for lbl, (r, c) in [("P1", PORT1), ("P2", PORT2)]:
        fig.add_trace(go.Scatter(
            x=[c], y=[r], mode="markers+text",
            marker=dict(size=14, color="cyan",
                        line=dict(width=2, color="#1a5276")),
            text=[lbl], textposition="top center",
            textfont=dict(size=9, color="#1a5276", family="monospace"),
            showlegend=False, hoverinfo="text",
            hovertext=f"Port {lbl[-1]} ({r},{c})",
        ))

    # Grid overlay (thin lines every pixel)
    for i in range(H + 1):
        fig.add_shape(type="line", x0=-0.5, x1=W - 0.5,
                      y0=i - 0.5, y1=i - 0.5,
                      line=dict(color="rgba(0,0,0,0.18)", width=0.5))
    for j in range(W + 1):
        fig.add_shape(type="line", x0=j - 0.5, x1=j - 0.5,
                      y0=-0.5, y1=H - 0.5,
                      line=dict(color="rgba(0,0,0,0.18)", width=0.5))

    fill_pct  = layout.mean() * 100
    connected = is_connected(layout)
    conn_col  = "#27ae60" if connected else "#e74c3c"
    conn_sym  = "✓" if connected else "✗"

    fig.add_annotation(
        text=f"Fill: {fill_pct:.1f}%  |  "
             f"<span style='color:{conn_col}'>{conn_sym} {'Connected' if connected else 'Disconnected'}</span>",
        xref="paper", yref="paper", x=0.5, y=-0.08,
        showarrow=False, font=dict(size=10), align="center",
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=12, family="monospace"), x=0.5),
        width=size, height=size + 30,
        margin=dict(l=10, r=10, t=35, b=40),
        xaxis=dict(showticklabels=False, showgrid=False,
                   zeroline=False, range=[-0.5, W - 0.5]),
        yaxis=dict(showticklabels=False, showgrid=False,
                   zeroline=False, range=[H - 0.5, -0.5],
                   scaleanchor="x"),
        plot_bgcolor="#f8f9fa",
    )
    return fig


def fig_sparams(
    target: np.ndarray,
    surrogate: dict | None = None,
    em: dict | None = None,
) -> go.Figure:
    """
    S-parameter comparison chart.
    Two subplots: S21 (left) and S11 (right), both in dB.
    """
    def db(mag):
        return 20 * np.log10(np.clip(np.abs(mag), 1e-6, 1.0))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["<b>S21</b> — Insertion Loss", "<b>S11</b> — Return Loss"],
        horizontal_spacing=0.10,
    )

    # ── Reference lines ──────────────────────────────────────────────────
    for col in [1, 2]:
        for lvl, lbl in [(-3, "-3 dB"), (-10, "-10 dB"), (-20, "-20 dB")]:
            fig.add_hline(y=lvl, line=dict(color="rgba(150,150,150,0.4)",
                                            dash="dot", width=1),
                          annotation_text=lbl,
                          annotation_font=dict(size=8, color="grey"),
                          row=1, col=col)

    # ── Target ───────────────────────────────────────────────────────────
    for col, ch in [(1, 1), (2, 0)]:
        fig.add_trace(go.Scatter(
            x=FREQS_GHZ, y=db(target[ch]),
            name="Target", legendgroup="target",
            showlegend=(col == 1),
            line=dict(color=_COLORS["target"], width=2, dash="dash"),
        ), row=1, col=col)

    # ── Surrogate ────────────────────────────────────────────────────────
    if surrogate is not None:
        for col, key in [(1, "s21_mag"), (2, "s11_mag")]:
            fig.add_trace(go.Scatter(
                x=FREQS_GHZ, y=db(surrogate[key]),
                name="Surrogate", legendgroup="surrogate",
                showlegend=(col == 1),
                line=dict(color=_COLORS["surrogate"], width=2),
            ), row=1, col=col)

    # ── EM ground truth ──────────────────────────────────────────────────
    if em is not None:
        s21 = np.array(em.get("s21_mag", em.get("S21_mag", [])))
        s11 = np.array(em.get("s11_mag", em.get("S11_mag", [])))
        if len(s21) == N_FREQ:
            for col, arr in [(1, s21), (2, s11)]:
                fig.add_trace(go.Scatter(
                    x=FREQS_GHZ, y=db(arr),
                    name="EM (OpenEMS)", legendgroup="em",
                    showlegend=(col == 1),
                    line=dict(color=_COLORS["em"], width=2.5),
                ), row=1, col=col)

    for col in [1, 2]:
        fig.update_xaxes(title_text="Frequency (GHz)", range=[0.5, 20],
                         row=1, col=col)
        fig.update_yaxes(title_text="Magnitude (dB)", range=[-45, 2],
                         row=1, col=col)

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.04,
                    xanchor="center", x=0.5,
                    font=dict(size=11)),
        margin=dict(l=55, r=20, t=65, b=50),
        height=360,
        font=dict(family="monospace"),
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="white",
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════════════════════════

def main():
    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown(
        "<h2 style='color:#1a5276;font-family:monospace;margin-bottom:0'>🔬 PIXEL</h2>"
        "<p style='color:#5d6d7e;margin-top:4px;font-size:0.9rem'>"
        "Physics-Guided Discrete Diffusion · Inverse EM Layout Synthesizer · PIXEL-2026</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Session state init ────────────────────────────────────────────────────
    # Do this ONCE before any conditional rendering so state persists across reruns
    for key, default in [
        ("layouts", []),
        ("surr_preds", []),
        ("y_star", None),
        ("em_result", None),
        ("spec_label", ""),
        ("substrate_name", "Rogers 4003C"),
        ("method", "PIXEL (Physics-Guided)"),
        ("generated", False),
        ("em_done", False),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Load models ───────────────────────────────────────────────────────────
    models, err = load_models()
    if models is None:
        st.error(f"**Model loading failed.**\n\n```\n{err}\n```")
        st.info("Run from project root with all Phase 2–4 checkpoints present.")
        return  # early return, not st.stop()

    device_str = str(models["device"])
    st.sidebar.success(f"✓ Models loaded · `{device_str}`")

    # ════════════════════════════════════════════════════════════════════════
    # SIDEBAR — all config in one form to avoid rerun-on-every-slider
    # ════════════════════════════════════════════════════════════════════════
    with st.sidebar:
        st.header("Target Specification")
        ftype = st.radio(
            "Filter type",
            ["Bandpass", "Bandstop", "Lowpass", "Wideband"],
            horizontal=False,
        )

        # Parameters conditional on filter type
        if ftype == "Bandpass":
            fc  = st.slider("Centre frequency (GHz)", 1.0, 18.0, 6.0, 0.5)
            bw  = st.slider("3 dB bandwidth (GHz)", 0.5, 8.0, 2.0, 0.5)
            y   = spec_bandpass(fc, bw)
            slbl = f"Bandpass  {fc:.1f} GHz ± {bw/2:.2f} GHz"

        elif ftype == "Bandstop":
            fc  = st.slider("Centre frequency (GHz)", 1.0, 18.0, 8.0, 0.5)
            bw  = st.slider("Rejection BW (GHz)", 0.5, 6.0, 1.5, 0.5)
            rej = st.slider("Stopband rejection (dB)", 10, 40, 20, 5)
            y   = spec_bandstop(fc, bw, rej)
            slbl = f"Bandstop  {fc:.1f} GHz / {rej} dB"

        elif ftype == "Lowpass":
            fc  = st.slider("Cutoff frequency (GHz)", 1.0, 15.0, 5.0, 0.5)
            y   = spec_lowpass(fc)
            slbl = f"Lowpass  {fc:.1f} GHz"

        else:
            y    = spec_wideband()
            slbl = "Wideband through"

        st.divider()
        st.header("Substrate")
        sub_name = st.selectbox("Material", list(SUBSTRATES.keys()), index=0)
        s_info   = SUBSTRATES[sub_name]
        st.caption(f"εr = {s_info['eps_r']}  ·  tanδ = {s_info['tan_d']}  ·  {s_info['note']}")

        st.divider()
        st.header("Generation Settings")
        method = st.radio(
            "Method",
            ["PIXEL (Physics-Guided)", "CFG Only", "No Guidance"],
            captions=[
                "Full surrogate + topology guidance",
                "Classifier-free guidance only",
                "Pure denoising (ablation)",
            ],
        )
        t_steps  = st.select_slider("Diffusion steps T",
                                     [100, 200, 500, 1000], value=1000)
        k_cands  = st.slider("K candidates", 1, 5, 1)
        cfg_w    = st.slider("CFG weight w", 0.0, 5.0, 2.0, 0.5)

        use_guidance = (method == "PIXEL (Physics-Guided)")
        eff_cfg_w    = 0.0 if method == "No Guidance" else cfg_w

        st.divider()
        gen_btn = st.button("▶  Generate Layout", type="primary",
                            use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Target specification preview (always visible)
    # ════════════════════════════════════════════════════════════════════════
    with st.expander("📊  Target Specification — " + slbl, expanded=True):
        c1, c2 = st.columns([4, 1])
        with c1:
            st.plotly_chart(fig_sparams(y), use_container_width=True,
                            key="chart_spec_target")
        with c2:
            st.markdown("**Spec summary**")
            s21_db_peak = 20 * np.log10(max(y[1].max(), 1e-6))
            s21_db_min  = 20 * np.log10(max(y[1].min(), 1e-6))
            st.metric("S21 peak",  f"{s21_db_peak:.1f} dB")
            st.metric("S21 floor", f"{s21_db_min:.1f} dB")
            st.metric("Substrate", sub_name)
            st.metric("Grid",      "15 × 15")

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Generation (triggers on button click)
    # ════════════════════════════════════════════════════════════════════════
    if gen_btn:
        # Reset EM state on new generation
        st.session_state.update({
            "y_star": y, "spec_label": slbl, "substrate_name": sub_name,
            "method": method, "layouts": [], "surr_preds": [],
            "em_result": None, "generated": False, "em_done": False,
        })

        layouts_out    = []
        surr_preds_out = []
        gen_t0 = time.time()

        for k in range(k_cands):
            with st.status(
                f"Generating candidate {k+1} / {k_cands}  ·  {method}  ·  T={t_steps}",
                expanded=True,
            ) as status:
                st.write(f"Seed: {k * 7919 + 42}  ·  {t_steps} denoising steps")
                pbar = st.progress(0)

                lay = generate(
                    y, models,
                    T=t_steps,
                    alpha_max=0.10 if use_guidance else 0.0,
                    cfg_w=eff_cfg_w,
                    use_guidance=use_guidance,
                    seed=k * 7919 + 42,
                    progress_bar=pbar,
                )
                surr = predict_surrogate(lay, models)
                layouts_out.append(lay)
                surr_preds_out.append(surr)

                m   = em_mse(y, surr["s21_mag"], surr["s11_mag"])
                cov = "✓" if m["Coverage @0.001"] else "○" if m["Coverage @0.010"] else "✗"
                status.update(
                    label=f"Candidate {k+1} complete  ·  "
                          f"Surrogate MSE: {m['Joint MSE']:.4f}  {cov}",
                    state="complete",
                )

        st.session_state.update({
            "layouts": layouts_out, "surr_preds": surr_preds_out, "generated": True,
        })
        gen_dur = time.time() - gen_t0
        st.success(f"Generated {k_cands} layout(s) in {gen_dur:.1f} s")

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 3 — Layout results (visible after generation)
    # ════════════════════════════════════════════════════════════════════════
    if st.session_state.generated and st.session_state.layouts:
        layouts    = st.session_state.layouts
        surr_preds = st.session_state.surr_preds
        y_star_st  = st.session_state.y_star

        st.divider()
        st.subheader(f"🏗️  Generated Layouts — {st.session_state.spec_label}")

        # Layout tiles (up to 5 across)
        cols = st.columns(min(len(layouts), 5))
        for k, (lay, surr) in enumerate(zip(layouts, surr_preds)):
            m    = em_mse(y_star_st, surr["s21_mag"], surr["s11_mag"])
            cov  = ("🟢" if m["Coverage @0.001"] else
                    "🟡" if m["Coverage @0.010"] else "🔴")
            with cols[k]:
                st.plotly_chart(fig_layout(lay, f"Candidate {k+1}"),
                                use_container_width=True,
                                key=f"chart_layout_tile_{k}")
                st.caption(
                    f"MSE: **{m['Joint MSE']:.4f}** {cov}\n\n"
                    f"σ̂: {surr['sigma']:.4f}"
                )

        # Best candidate by surrogate MSE
        best_k = int(np.argmin([
            em_mse(y_star_st, s["s21_mag"], s["s11_mag"])["Joint MSE"]
            for s in surr_preds
        ]))

        # S-parameter comparison: best candidate vs target
        st.subheader("📈  Surrogate Prediction vs Target")
        st.plotly_chart(
            fig_sparams(y_star_st, surrogate=surr_preds[best_k]),
            use_container_width=True,
            key="chart_surr_best",
        )
        if len(layouts) > 1:
            st.caption(
                f"Showing best candidate (Candidate {best_k+1}) by surrogate MSE. "
                "Select a specific candidate in EM Verification below."
            )

        # ════════════════════════════════════════════════════════════════════
        # SECTION 4 — EM Verification
        # ════════════════════════════════════════════════════════════════════
        st.divider()
        st.subheader("⚡  EM Ground-Truth Verification")
        st.caption(
            "Runs a full OpenEMS FDTD simulation (~18–28 s) to obtain ground-truth "
            "S-parameters independent of the surrogate model."
        )

        cA, cB = st.columns([2, 1])
        with cA:
            sel_k = st.selectbox(
                "Candidate to verify",
                options=list(range(len(layouts))),
                format_func=lambda k: f"Candidate {k+1}",
                index=best_k,
                key="em_sel_k",
            )
        with cB:
            st.markdown("<br>", unsafe_allow_html=True)
            run_em_btn = st.button("⚡  Run OpenEMS",
                                   type="primary", use_container_width=True)

        if run_em_btn:
            st.session_state.em_result = None
            st.session_state.em_done   = False
            with st.spinner(
                f"Running OpenEMS FDTD on {st.session_state.substrate_name}  "
                "(18–28 seconds)…"
            ):
                t0  = time.time()
                res = run_em(layouts[sel_k], s_info["id"])
                dur = time.time() - t0

            if res is not None:
                st.session_state.em_result = res
                st.session_state.em_done   = True
                st.success(f"EM simulation complete in {dur:.1f} s")

        # Show EM results if available
        if st.session_state.em_done and st.session_state.em_result is not None:
            em_res    = st.session_state.em_result
            sel_lay   = layouts[st.session_state.get("em_sel_k", best_k)]
            sel_surr  = surr_preds[st.session_state.get("em_sel_k", best_k)]

            em_s21 = np.array(em_res.get("s21_mag", em_res.get("S21_mag", [])))
            em_s11 = np.array(em_res.get("s11_mag", em_res.get("S11_mag", [])))

            # Full 3-way comparison
            st.subheader("📊  Target vs Surrogate vs EM Ground Truth")
            st.plotly_chart(
                fig_sparams(y_star_st, surrogate=sel_surr, em=em_res),
                use_container_width=True,
                key="chart_em_comparison",
            )

            # Metrics side-by-side
            st.markdown("#### Performance Metrics")
            m_surr = em_mse(y_star_st, sel_surr["s21_mag"], sel_surr["s11_mag"])
            m_em   = em_mse(y_star_st, em_s21, em_s11) if len(em_s21) == N_FREQ \
                     else {}

            cL, cM, cR = st.columns(3)

            with cL:
                st.markdown("**Surrogate**")
                for k, v in m_surr.items():
                    if "Coverage" in k:
                        st.metric(k, "✓ Yes" if v else "✗ No")
                    else:
                        st.metric(k, f"{v:.5f}")

            with cM:
                st.markdown("**EM Ground Truth**")
                if m_em:
                    for k, v in m_em.items():
                        if "Coverage" in k:
                            delta = None
                            if k in m_surr:
                                delta_v = int(v) - int(m_surr[k])
                                delta = "+Improved" if delta_v > 0 else \
                                        ("Degraded" if delta_v < 0 else "Same")
                            st.metric(k, "✓ Yes" if v else "✗ No", delta=delta)
                        else:
                            st.metric(k, f"{v:.5f}")
                else:
                    st.info("EM array length mismatch")

            with cR:
                st.markdown("**Layout & Physics**")
                st.metric("Fill fraction",
                          f"{sel_lay.mean()*100:.1f}%")
                st.metric("Connected",
                          "Yes" if is_connected(sel_lay) else "No")
                passivity = em_res.get("passivity_ok",
                            em_res.get("passivity_OK", True))
                st.metric("Passivity OK", "Yes" if passivity else "No")
                kk_res = em_res.get("kk_residual", float("nan"))
                st.metric("KK residual", f"{kk_res:.3f}")

            # Surrogate accuracy vs EM
            if len(em_s21) == N_FREQ:
                surr_em = float(np.mean(
                    (sel_surr["s21_mag"] - em_s21) ** 2
                    + (sel_surr["s11_mag"] - em_s11) ** 2
                ) / 2)
                st.metric("Surrogate–EM MSE",
                          f"{surr_em:.5f}",
                          help="How accurately the surrogate predicted the actual EM response")

            # ════════════════════════════════════════════════════════════════
            # SECTION 5 — Gerber Export
            # ════════════════════════════════════════════════════════════════
            st.divider()
            st.subheader("📦  Export — Fabrication-Ready Gerber Package")
            st.markdown(
                "Download a complete Gerber + drill package for PCB fabrication. "
                "Includes copper top, board outline, soldermask, drill file, "
                "and full metrics."
            )

            col_info, col_btn = st.columns([3, 1])
            with col_info:
                st.markdown(
                    f"- **Board size:** {W * PIXEL_MM:.1f} mm × {H * PIXEL_MM:.1f} mm  \n"
                    f"- **Conductor pixels:** {int(sel_lay.sum())} / {H*W}  \n"
                    f"- **Substrate:** {st.session_state.substrate_name}  \n"
                    f"- **EM MSE:** {m_em.get('Joint MSE', float('nan')):.5f}  \n"
                    f"- **Verified:** Full-wave OpenEMS FDTD ✓"
                )
            with col_btn:
                st.markdown("<br>", unsafe_allow_html=True)
                gerber_bytes = make_gerber_zip(
                    layout=sel_lay,
                    spec_label=st.session_state.spec_label,
                    substrate=st.session_state.substrate_name,
                    method=st.session_state.method,
                    surr_metrics=m_surr,
                    em_metrics=m_em if m_em else None,
                )
                safe_label = st.session_state.spec_label.replace(" ", "_").replace("/", "-")
                st.download_button(
                    label="📥  Download Gerber (.zip)",
                    data=gerber_bytes,
                    file_name=f"PIXEL_{safe_label}.zip",
                    mime="application/zip",
                    use_container_width=True,
                    type="primary",
                )

        elif not st.session_state.em_done:
            # No EM yet — show surrogate comparison for selected candidate
            sel_surr_cur = surr_preds[best_k]
            st.info(
                "Click **⚡ Run OpenEMS** above to run a full-wave ground-truth "
                "simulation and unlock the **Gerber export**."
            )
            st.plotly_chart(
                fig_sparams(y_star_st, surrogate=sel_surr_cur),
                use_container_width=True,
                key="chart_surr_preview",
            )

    elif not st.session_state.generated:
        # First visit — show instructions
        st.markdown("""
---
### How to use

1. **Set the target response** in the sidebar — choose a filter type and adjust the frequency parameters.
2. **Select a substrate** material.
3. **Choose a generation method:**
   - *PIXEL (Physics-Guided)* — full surrogate + topology guidance (recommended)
   - *CFG Only* — classifier-free guidance only
   - *No Guidance* — ablation: pure denoising
4. **Click ▶ Generate Layout** — the diffusion model generates a 15×15 binary RF layout.
5. **Review** the layout and surrogate S-parameter prediction.
6. **Click ⚡ Run OpenEMS** to verify with full-wave FDTD simulation.
7. **Download** the fabrication-ready **Gerber package** (.zip).

---
> **PIXEL-2026** · Physics-Constrained Probabilistic Topology Synthesis for Inverse EM Design
> 342,415-sample dataset · D3PM absorbing diffusion · K=5 surrogate ensemble · OpenEMS FDTD verification
""")


if __name__ == "__main__":
    main()
