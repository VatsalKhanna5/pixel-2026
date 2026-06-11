"""
app/pixel_demo.py
=================
PIXEL — Physics-Guided Inverse EM Layout Synthesizer
Interactive Streamlit demo for PIXEL-2026.

Run from project root:
    streamlit run app/pixel_demo.py --server.address 0.0.0.0 --server.port 8501

Remote access from local PC:
    ssh -L 8501:localhost:8501 ec_23104075@10.10.11.201
    then open http://localhost:8501 in your browser
"""

from __future__ import annotations

import sys
import os
import time
import subprocess
import tempfile
import traceback
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Project imports ─────────────────────────────────────────────────────────
from src.models.surrogate import PhysicsSurrogate, SurrogateEnsemble
from src.models.denoiser import PixelDenoiser
from src.models.spectral_encoder import SpectralEncoder
from src.models.connectivity_disc import ConnectivityDiscriminator
from src.models.diffusion import D3PMAbsorbing
from src.guidance.physics_guidance import guided_reverse_step
from src.guidance.cfg import discrete_cfg
from src.dataset.connectivity import is_connected, PORT1, PORT2

# ── Constants ────────────────────────────────────────────────────────────────
FREQS_GHZ  = np.linspace(0.5, 20.0, 100)           # frequency axis [GHz]
FREQS_HZ   = FREQS_GHZ * 1e9                        # [Hz] for OpenEMS
N_FREQ     = 100
H, W       = 15, 15
PORT_MAP   = np.zeros((H, W), dtype=np.float32)
PORT_MAP[PORT1[0], PORT1[1]] = 1.0
PORT_MAP[PORT2[0], PORT2[1]] = 1.0

SURROGATE_PATHS = [
    f"experiments/surrogate_v1/surrogate_k{k}_best.pt" for k in range(5)
]
DENOISER_PATH    = "experiments/denoiser_v1/denoiser_best.pt"
DISC_PATH        = "experiments/discriminator_v1/disc_best.pt"

SUBSTRATE_IDS = {
    "Rogers4003C": 0,
    "FR4":         1,
    "Rogers5880":  2,
    "Alumina":     3,
}

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PIXEL — Inverse EM Synthesizer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-header {
    font-size: 1.8rem; font-weight: 700;
    background: linear-gradient(90deg, #1f4e79, #2980b9);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.metric-card {
    background: #f0f4f8; border-radius: 8px; padding: 12px;
    border-left: 4px solid #2980b9;
}
.good-metric { border-left-color: #27ae60 !important; }
.bad-metric  { border-left-color: #e74c3c !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Model loading — cached so models are loaded only once per session
# ============================================================================

@st.cache_resource(show_spinner="Loading PIXEL models...")
def load_models():
    """Load all trained models from checkpoints. Cached across reruns."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    missing = [p for p in SURROGATE_PATHS + [DENOISER_PATH, DISC_PATH]
               if not Path(p).exists()]
    if missing:
        return None, f"Missing checkpoints:\n" + "\n".join(missing)

    try:
        # Surrogate ensemble
        surrogate_ens = SurrogateEnsemble.load(SURROGATE_PATHS, device=device)
        surrogate_ens.eval()

        # Denoiser + encoder
        ckpt = torch.load(DENOISER_PATH, map_location=device, weights_only=False)
        denoiser = PixelDenoiser()
        encoder  = SpectralEncoder()
        denoiser.load_state_dict(ckpt["ema_state"])   # use EMA weights
        encoder.load_state_dict(ckpt["encoder_state"])
        denoiser.eval().to(device)
        encoder.eval().to(device)

        # Connectivity discriminator
        ckpt_d = torch.load(DISC_PATH, map_location=device, weights_only=False)
        disc = ConnectivityDiscriminator()
        disc.load_state_dict(ckpt_d["model_state"])
        disc.eval().to(device)

        # D3PM absorbing-state diffusion
        diffusion = D3PMAbsorbing(T=1000)

        # Port map tensor
        pm_tensor = torch.from_numpy(PORT_MAP).unsqueeze(0).unsqueeze(0).to(device)

        models = {
            "surrogate":     surrogate_ens,
            "denoiser":      denoiser,
            "encoder":       encoder,
            "discriminator": disc,
            "diffusion":     diffusion,
            "port_map":      pm_tensor,
            "device":        device,
        }
        return models, None

    except Exception as e:
        return None, f"Failed to load models: {e}\n{traceback.format_exc()}"


# ============================================================================
# Specification builders
# ============================================================================

def build_bandpass_spec(f_center_ghz: float, bw_ghz: float) -> np.ndarray:
    """Gaussian bandpass centred at f_center with 3dB BW = bw_ghz."""
    fc  = f_center_ghz * 1e9
    bw  = bw_ghz * 1e9
    s21 = np.exp(-np.log(2) * ((FREQS_HZ - fc) / (bw / 2)) ** 2)
    s21 = np.clip(s21, 1e-4, 1.0)
    s11 = np.sqrt(np.clip(1.0 - s21 ** 2, 1e-4, 1.0))
    gd  = 1.0 / (fc * 1e-9 * 2 * np.pi) * 0.5   # simple group delay
    s21_phase = -2 * np.pi * FREQS_HZ * gd
    s11_phase = s21_phase + np.pi
    return _pack_spec(s11, s21, s11_phase, s21_phase)


def build_bandstop_spec(f_center_ghz: float, bw_ghz: float,
                        rejection_db: float) -> np.ndarray:
    """Gaussian bandstop with rejection_db attenuation at f_center."""
    fc  = f_center_ghz * 1e9
    bw  = bw_ghz * 1e9
    att = 10 ** (-abs(rejection_db) / 20)          # linear attenuation floor
    s21 = 1.0 - (1.0 - att) * np.exp(-np.log(2) * ((FREQS_HZ - fc) / (bw / 2)) ** 2)
    s21 = np.clip(s21, att, 1.0)
    s11 = np.sqrt(np.clip(1.0 - s21 ** 2, 1e-4, 1.0))
    gd  = 0.3e-9
    s21_phase = -2 * np.pi * FREQS_HZ * gd
    s11_phase = s21_phase + np.pi
    return _pack_spec(s11, s21, s11_phase, s21_phase)


def build_lowpass_spec(f_cutoff_ghz: float, order: int = 3) -> np.ndarray:
    """Butterworth-like lowpass with cutoff f_cutoff_ghz."""
    fc  = f_cutoff_ghz * 1e9
    s21 = 1.0 / np.sqrt(1.0 + (FREQS_HZ / fc) ** (2 * order))
    s21 = np.clip(s21, 1e-4, 1.0)
    s11 = np.sqrt(np.clip(1.0 - s21 ** 2, 1e-4, 1.0))
    gd  = 0.3e-9
    s21_phase = -2 * np.pi * FREQS_HZ * gd
    s11_phase = s21_phase + np.pi
    return _pack_spec(s11, s21, s11_phase, s21_phase)


def build_wideband_spec() -> np.ndarray:
    """Flat wideband through-connection."""
    s21 = np.ones(N_FREQ) * 0.92
    s11 = np.ones(N_FREQ) * 0.18
    gd  = 0.2e-9
    s21_phase = -2 * np.pi * FREQS_HZ * gd
    s11_phase = s21_phase + np.pi
    return _pack_spec(s11, s21, s11_phase, s21_phase)


def _pack_spec(s11_mag, s21_mag, s11_ph, s21_ph) -> np.ndarray:
    """Pack and normalise into (4, 100) model input format."""
    # Wrap phases to [-π, π]
    s11_ph = (s11_ph + np.pi) % (2 * np.pi) - np.pi
    s21_ph = (s21_ph + np.pi) % (2 * np.pi) - np.pi
    return np.stack([
        s11_mag.astype(np.float32),
        s21_mag.astype(np.float32),
        (s11_ph / np.pi).astype(np.float32),
        (s21_ph / np.pi).astype(np.float32),
    ], axis=0)   # (4, 100)


# ============================================================================
# Layout utilities
# ============================================================================

def remove_floating_islands(layout: np.ndarray) -> np.ndarray:
    """BFS: keep only conductors reachable from either port pixel."""
    from collections import deque
    layout = layout.copy()
    reachable = set()
    for start in [PORT1, PORT2]:
        if layout[start] == 0:
            continue
        q = deque([start])
        visited = {start}
        while q:
            r, c = q.popleft()
            reachable.add((r, c))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < H and 0 <= nc < W and layout[nr,nc]==1 \
                        and (nr,nc) not in visited:
                    visited.add((nr,nc))
                    q.append((nr,nc))
    for r in range(H):
        for c in range(W):
            if layout[r,c] == 1 and (r,c) not in reachable:
                layout[r,c] = 0
    return layout


def surrogate_predict(layout: np.ndarray, models: dict) -> dict:
    """Run surrogate ensemble on a single binary layout."""
    device = models["device"]
    pm     = PORT_MAP
    x_inp  = np.stack([layout.astype(np.float32), pm], axis=0)[None]  # (1,2,15,15)
    x_t    = torch.from_numpy(x_inp).to(device)
    with torch.no_grad():
        mean, var = models["surrogate"](x_t)
    mean = mean.cpu().numpy()[0]   # (4, 100)
    var  = var.cpu().numpy()[0]
    return {
        "s11_mag":   mean[0],
        "s21_mag":   mean[1],
        "s11_phase": mean[2] * np.pi,
        "s21_phase": mean[3] * np.pi,
        "uncertainty": float(np.sqrt(var.mean())),
    }


# ============================================================================
# Generation with real-time progress
# ============================================================================

def generate_layout(
    y_star_np: np.ndarray,
    models: dict,
    T: int = 1000,
    t_thresh: int = 400,
    alpha_max: float = 0.10,
    cfg_w: float = 2.0,
    use_guidance: bool = True,
    progress_bar=None,
    status_text=None,
) -> np.ndarray:
    """
    Run one full reverse diffusion pass and return a cleaned binary layout.

    Args:
        y_star_np: (4, 100) target S-params (normalised, phases÷π)
    Returns:
        layout: (15, 15) binary numpy array
    """
    device   = models["device"]
    denoiser = models["denoiser"]
    encoder  = models["encoder"]
    diffusion = models["diffusion"]
    surrogate_ens = models["surrogate"]
    discriminator = models["discriminator"]
    pm = models["port_map"]

    y_star = torch.from_numpy(y_star_np)[None].to(device)   # (1, 4, 100)

    with torch.no_grad():
        c_y = encoder(y_star)                                # (1, 256)

    x_t = torch.full((1, H, W), diffusion.MASK, dtype=torch.long, device=device)

    effective_thresh = t_thresh if use_guidance else 0

    for idx, t_val in enumerate(range(T, 0, -1)):
        t_tensor = torch.full((1,), t_val, dtype=torch.long, device=device)
        x_t = guided_reverse_step(
            x_t, t_val, t_tensor,
            denoiser, encoder, diffusion, surrogate_ens, discriminator,
            pm, c_y, y_star,
            T=T,
            t_thresh=effective_thresh,
            alpha_max=alpha_max,
            epsilon=0.01,
            lambda_topo=1.0,
            lambda_mfg=0.5,
            g_max=1.0,
            cfg_w=cfg_w,
        )

        # Update progress every 50 steps
        if progress_bar is not None and idx % 50 == 0:
            progress_bar.progress((idx + 1) / T,
                                  text=f"Denoising step {idx+1}/{T}...")

    if progress_bar is not None:
        progress_bar.progress(1.0, text="Generation complete!")

    layout = x_t[0].cpu().numpy().astype(np.uint8)
    return remove_floating_islands(layout)


# ============================================================================
# OpenEMS verification
# ============================================================================

def run_openems(layout: np.ndarray, substrate_id: int = 0) -> dict | None:
    """
    Run a full OpenEMS FDTD simulation on the given layout.
    Calls the existing openems_wrapper.simulate() function.
    Returns the results dict or None on failure.
    """
    try:
        from src.dataset.openems_wrapper import simulate
        result = simulate(layout, substrate_id=substrate_id)
        return result
    except Exception as e:
        st.error(f"OpenEMS simulation failed: {e}")
        return None


# ============================================================================
# Plotting helpers
# ============================================================================

def plot_layout(layout: np.ndarray, title: str = "Generated Layout") -> go.Figure:
    """
    Render a 15x15 binary layout as a heatmap.
    Conductor (1) = copper gold, Dielectric (0) = light grey.
    Port pixels marked with cyan dots.
    """
    fig = go.Figure()

    # Background heatmap
    colorscale = [[0, "#f8f9fa"], [1, "#b87333"]]    # white → copper
    fig.add_trace(go.Heatmap(
        z=layout.astype(float),
        colorscale=colorscale,
        zmin=0, zmax=1,
        showscale=False,
        hovertemplate="row=%{y}  col=%{x}  value=%{z}<extra></extra>",
    ))

    # Port markers
    for label, (r, c) in [("P1", PORT1), ("P2", PORT2)]:
        fig.add_trace(go.Scatter(
            x=[c], y=[r], mode="markers+text",
            marker=dict(size=12, color="cyan", symbol="circle",
                        line=dict(width=2, color="navy")),
            text=[label], textposition="top center",
            textfont=dict(size=10, color="navy"),
            showlegend=False,
            hoverinfo="text", hovertext=f"{label} port",
        ))

    # Grid lines
    for i in range(H + 1):
        fig.add_shape(type="line", x0=-0.5, x1=W-0.5, y0=i-0.5, y1=i-0.5,
                      line=dict(color="rgba(100,100,100,0.3)", width=0.5))
    for j in range(W + 1):
        fig.add_shape(type="line", x0=j-0.5, x1=j-0.5, y0=-0.5, y1=H-0.5,
                      line=dict(color="rgba(100,100,100,0.3)", width=0.5))

    # Metrics annotation
    fill_frac = layout.mean() * 100
    connected = is_connected(layout)
    conn_str  = "✓ Connected" if connected else "✗ Disconnected"
    fig.add_annotation(
        text=f"Fill: {fill_frac:.1f}%  |  {conn_str}",
        xref="paper", yref="paper", x=0.5, y=-0.08,
        showarrow=False, font=dict(size=11),
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        width=300, height=320,
        margin=dict(l=30, r=30, t=40, b=40),
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False,
                   autorange="reversed"),
        plot_bgcolor="white",
    )
    return fig


def plot_sparams(
    target: np.ndarray,
    surrogate: dict | None = None,
    em_result: dict | None = None,
    title: str = "S-Parameter Comparison",
) -> go.Figure:
    """
    Plot S21 and S11 magnitude (in dB) vs frequency.
    target:    (4,100) array [s11_mag, s21_mag, s11_ph/π, s21_ph/π]
    surrogate: dict with s11_mag, s21_mag keys
    em_result: dict from openems_wrapper.simulate()
    """
    freqs = FREQS_GHZ

    def to_db(mag):
        return 20 * np.log10(np.clip(np.abs(mag), 1e-6, 1.0))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["S21 — Insertion Loss (dB)", "S11 — Return Loss (dB)"],
        shared_xaxes=False,
    )

    # ── Target ──────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=freqs, y=to_db(target[1]),
        mode="lines", name="Target S21",
        line=dict(color="#e74c3c", width=2, dash="dash"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=freqs, y=to_db(target[0]),
        mode="lines", name="Target S11",
        line=dict(color="#e74c3c", width=2, dash="dash"),
    ), row=1, col=2)

    # ── Surrogate prediction ─────────────────────────────────────────────────
    if surrogate is not None:
        fig.add_trace(go.Scatter(
            x=freqs, y=to_db(surrogate["s21_mag"]),
            mode="lines", name="Surrogate S21",
            line=dict(color="#2980b9", width=2),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=freqs, y=to_db(surrogate["s11_mag"]),
            mode="lines", name="Surrogate S11",
            line=dict(color="#2980b9", width=2),
        ), row=1, col=2)

    # ── OpenEMS ground truth ─────────────────────────────────────────────────
    if em_result is not None:
        fig.add_trace(go.Scatter(
            x=freqs, y=to_db(em_result.get("s21_mag", em_result.get("S21_mag", []))),
            mode="lines", name="EM (OpenEMS) S21",
            line=dict(color="#27ae60", width=2.5),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=freqs, y=to_db(em_result.get("s11_mag", em_result.get("S11_mag", []))),
            mode="lines", name="EM (OpenEMS) S11",
            line=dict(color="#27ae60", width=2.5),
        ), row=1, col=2)

    for col in [1, 2]:
        fig.update_xaxes(title_text="Frequency (GHz)", row=1, col=col)
        fig.update_yaxes(title_text="Magnitude (dB)", range=[-40, 1], row=1, col=col)

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=380, margin=dict(l=50, r=20, t=60, b=50),
    )
    return fig


def compute_mse(target: np.ndarray, pred_mag_s21: np.ndarray,
                pred_mag_s11: np.ndarray) -> dict:
    """Compute MSE and coverage metrics."""
    s21_mse = float(np.mean((pred_mag_s21 - target[1]) ** 2))
    s11_mse = float(np.mean((pred_mag_s11 - target[0]) ** 2))
    joint_mse = (s21_mse + s11_mse) / 2
    cov_001 = float(joint_mse < 0.001)
    cov_010 = float(joint_mse < 0.010)
    return {
        "S21 MSE": s21_mse,
        "S11 MSE": s11_mse,
        "Joint MSE": joint_mse,
        "Coverage @0.001": cov_001,
        "Coverage @0.010": cov_010,
    }


# ============================================================================
# Main app
# ============================================================================

def main():
    st.markdown('<div class="main-header">🔬 PIXEL — Inverse EM Layout Synthesizer</div>',
                unsafe_allow_html=True)
    st.caption("Physics-Guided Discrete Diffusion for RF/IC Electromagnetic Inverse Design · PIXEL-2026")
    st.divider()

    # ── Load models ──────────────────────────────────────────────────────────
    models, err = load_models()
    if models is None:
        st.error(f"**Model loading failed.** {err}")
        st.info("Make sure you are running from the project root and all Phase 2–4 "
                "checkpoints are present in `experiments/`.")
        st.stop()

    device_str = str(models["device"])
    st.sidebar.success(f"Models loaded · device: `{device_str}`")

    # ── Sidebar: S-parameter specification ───────────────────────────────────
    st.sidebar.header("1. Target S-Parameter Spec")
    filter_type = st.sidebar.radio(
        "Filter type",
        ["Bandpass", "Bandstop", "Lowpass", "Wideband"],
        index=0,
    )

    y_star_np: np.ndarray | None = None

    if filter_type == "Bandpass":
        f_center = st.sidebar.slider("Centre frequency (GHz)", 1.0, 18.0, 6.0, 0.5)
        bw       = st.sidebar.slider("3dB bandwidth (GHz)", 0.5, 8.0, 2.0, 0.5)
        y_star_np = build_bandpass_spec(f_center, bw)
        spec_label = f"Bandpass · {f_center:.1f} GHz ± {bw/2:.1f} GHz"

    elif filter_type == "Bandstop":
        f_center  = st.sidebar.slider("Centre frequency (GHz)", 1.0, 18.0, 8.0, 0.5)
        bw        = st.sidebar.slider("Rejection bandwidth (GHz)", 0.5, 6.0, 1.5, 0.5)
        rejection = st.sidebar.slider("Stopband rejection (dB)", 10, 40, 20, 5)
        y_star_np = build_bandstop_spec(f_center, bw, rejection)
        spec_label = f"Bandstop · {f_center:.1f} GHz, {rejection} dB rejection"

    elif filter_type == "Lowpass":
        f_cutoff = st.sidebar.slider("Cutoff frequency (GHz)", 1.0, 15.0, 5.0, 0.5)
        y_star_np = build_lowpass_spec(f_cutoff)
        spec_label = f"Lowpass · {f_cutoff:.1f} GHz cutoff"

    else:  # Wideband
        y_star_np = build_wideband_spec()
        spec_label = "Wideband through-connection"

    st.sidebar.divider()

    # ── Sidebar: Substrate ────────────────────────────────────────────────────
    st.sidebar.header("2. Substrate")
    substrate_name = st.sidebar.selectbox(
        "Substrate material",
        list(SUBSTRATE_IDS.keys()),
        index=0,
    )
    substrate_info = {
        "Rogers4003C": "εr=3.55, tanδ=0.0027 — High-freq standard",
        "FR4":         "εr=4.4, tanδ=0.02 — Low-cost PCB",
        "Rogers5880":  "εr=2.2, tanδ=0.0009 — mmWave/PTFE",
        "Alumina":     "εr=9.8, tanδ=0.0001 — MMIC/ceramic",
    }
    st.sidebar.caption(substrate_info[substrate_name])
    substrate_id = SUBSTRATE_IDS[substrate_name]

    st.sidebar.divider()

    # ── Sidebar: Generation settings ─────────────────────────────────────────
    st.sidebar.header("3. Generation Settings")
    method = st.sidebar.radio(
        "Generation method",
        ["PIXEL (Physics-Guided)", "CFG Only", "No Guidance (Ablation)"],
        index=0,
        help="PIXEL uses physics guidance + CFG. CFG uses only classifier-free guidance. No Guidance = pure denoising.",
    )
    T_steps = st.sidebar.select_slider(
        "Diffusion steps (T)",
        options=[100, 200, 500, 1000],
        value=1000,
        help="Full T=1000 gives best quality. T=100 is 10× faster for quick exploration.",
    )
    k_candidates = st.sidebar.slider("K candidates", 1, 5, 1,
                                      help="Generate K diverse layouts for this spec")
    cfg_w = st.sidebar.slider("CFG guidance weight (w)", 0.0, 5.0, 2.0, 0.5,
                               help="Higher = more faithful to spec, less diversity")

    use_guidance = method == "PIXEL (Physics-Guided)"
    cfg_w_eff    = 0.0 if method == "No Guidance (Ablation)" else cfg_w

    generate_btn = st.sidebar.button("▶ Generate Layout", type="primary",
                                      use_container_width=True)

    # ── Main panel: Tabs ──────────────────────────────────────────────────────
    tab_spec, tab_gen, tab_em = st.tabs([
        "📊 Target Specification",
        "🏗️ Generated Layouts",
        "⚡ EM Verification",
    ])

    # ── Tab 1: Target Specification ───────────────────────────────────────────
    with tab_spec:
        st.subheader(f"Target Response: {spec_label}")
        col_plot, col_info = st.columns([3, 1])
        with col_plot:
            fig_spec = plot_sparams(y_star_np, title="Target S-Parameter Specification")
            st.plotly_chart(fig_spec, use_container_width=True)
        with col_info:
            st.markdown("**Spec summary**")
            s21_db_peak = float(20 * np.log10(max(y_star_np[1].max(), 1e-6)))
            s21_db_min  = float(20 * np.log10(max(y_star_np[1].min(), 1e-6)))
            st.metric("S21 peak (dB)",  f"{s21_db_peak:.1f}")
            st.metric("S21 floor (dB)", f"{s21_db_min:.1f}")
            st.metric("Freq range",     f"0.5–20 GHz")
            st.metric("Grid",           "15×15 = 225 pixels")
            st.caption(
                f"Substrate: {substrate_name}  \n"
                f"Port 1: row 7 col 0  \n"
                f"Port 2: row 7 col 14"
            )
            if y_star_np is not None:
                spec_bytes = y_star_np.tobytes()
                st.download_button(
                    "Download spec (binary)",
                    data=spec_bytes,
                    file_name="target_spec.bin",
                    mime="application/octet-stream",
                )

    # ── Tab 2: Generated Layouts ───────────────────────────────────────────────
    with tab_gen:
        if not generate_btn:
            st.info("Configure your target specification in the sidebar, then click "
                    "**▶ Generate Layout**.")
            st.stop()

        st.subheader(f"Generating {k_candidates} layout(s) · {method} · T={T_steps}")

        # Store results in session state so Tab 3 can access them
        if "layouts" not in st.session_state:
            st.session_state.layouts     = []
            st.session_state.surr_preds  = []
            st.session_state.y_star      = None
            st.session_state.em_result   = None
            st.session_state.selected_k  = 0

        st.session_state.y_star    = y_star_np
        st.session_state.em_result = None
        layouts     = []
        surr_preds  = []

        gen_start = time.time()
        for k in range(k_candidates):
            st.markdown(f"**Candidate {k+1}/{k_candidates}**")
            pbar   = st.progress(0, text="Starting denoising chain...")
            status = st.empty()

            # Seed per candidate
            torch.manual_seed(k * 1000 + 42)
            np.random.seed(k * 1000 + 42)

            layout = generate_layout(
                y_star_np, models,
                T=T_steps,
                t_thresh=int(T_steps * 0.4),
                alpha_max=0.10 if use_guidance else 0.0,
                cfg_w=cfg_w_eff,
                use_guidance=use_guidance,
                progress_bar=pbar,
                status_text=status,
            )
            surr = surrogate_predict(layout, models)

            layouts.append(layout)
            surr_preds.append(surr)
            status.empty()

        gen_time = time.time() - gen_start
        st.success(f"Generated {k_candidates} layout(s) in {gen_time:.1f}s")

        st.session_state.layouts    = layouts
        st.session_state.surr_preds = surr_preds

        # Display all candidates
        cols = st.columns(min(k_candidates, 5))
        for k, (layout, surr) in enumerate(zip(layouts, surr_preds)):
            with cols[k % 5]:
                fig_lay = plot_layout(layout, title=f"Candidate {k+1}")
                st.plotly_chart(fig_lay, use_container_width=True)

                mse = compute_mse(y_star_np, surr["s21_mag"], surr["s11_mag"])
                st.caption(
                    f"Surrogate MSE: **{mse['Joint MSE']:.4f}**  \n"
                    f"Uncertainty σ̂: {surr['uncertainty']:.4f}  \n"
                    f"Connected: {'✓' if is_connected(layout) else '✗'}"
                )

        # S-param comparison: all candidates vs target
        st.subheader("Surrogate Prediction vs Target")
        fig_cmp = plot_sparams(y_star_np,
                               surrogate=surr_preds[0] if surr_preds else None,
                               title=f"Candidate 1 · Surrogate vs Target")
        st.plotly_chart(fig_cmp, use_container_width=True)

        if k_candidates > 1:
            st.caption("Showing Candidate 1 surrogate prediction. Select a candidate "
                       "in the EM Verification tab to run ground-truth simulation.")

        # Best-of-K selection
        if k_candidates > 1:
            mses = [compute_mse(y_star_np, s["s21_mag"], s["s11_mag"])["Joint MSE"]
                    for s in surr_preds]
            best_k = int(np.argmin(mses))
            st.info(f"Best candidate by surrogate MSE: **Candidate {best_k+1}** "
                    f"(MSE={mses[best_k]:.4f})")
            st.session_state.selected_k = best_k

    # ── Tab 3: EM Verification ─────────────────────────────────────────────────
    with tab_em:
        layouts    = st.session_state.get("layouts", [])
        surr_preds = st.session_state.get("surr_preds", [])
        y_star_st  = st.session_state.get("y_star", y_star_np)

        if not layouts:
            st.info("Generate a layout first (Tab 2), then run EM verification here.")
            st.stop()

        st.subheader("Full-Wave EM Verification (OpenEMS FDTD)")
        st.caption(
            "Runs a real OpenEMS FDTD simulation (~18–28 seconds) to obtain ground-truth "
            "S-parameters independent of the surrogate model."
        )

        col_sel, col_btn = st.columns([2, 1])
        with col_sel:
            sel_k = st.selectbox(
                "Select candidate to verify",
                options=list(range(len(layouts))),
                format_func=lambda k: f"Candidate {k+1}",
                index=st.session_state.get("selected_k", 0),
            )
        with col_btn:
            st.write("")  # vertical spacing
            run_em_btn = st.button("⚡ Run OpenEMS Verification",
                                   type="primary", use_container_width=True)

        selected_layout = layouts[sel_k]
        selected_surr   = surr_preds[sel_k]

        # Layout preview
        st.plotly_chart(plot_layout(selected_layout, f"Candidate {sel_k+1} — Selected"),
                        use_container_width=False)

        if run_em_btn:
            with st.spinner(
                f"Running OpenEMS FDTD simulation on {substrate_name}... (~20-30 seconds)"
            ):
                t0 = time.time()
                em = run_openems(selected_layout, substrate_id=substrate_id)
                em_time = time.time() - t0

            if em is None:
                st.error("OpenEMS simulation failed. Check that OpenEMS is installed "
                         "and the layout is valid.")
            else:
                st.session_state.em_result = em
                st.success(f"EM simulation complete in {em_time:.1f}s")

        em_result = st.session_state.get("em_result")

        if em_result is not None:
            # Full comparison plot
            st.subheader("Target vs Surrogate vs EM Ground Truth")
            fig_full = plot_sparams(
                y_star_st,
                surrogate=selected_surr,
                em_result=em_result,
                title="S-Parameter Comparison · All Sources",
            )
            st.plotly_chart(fig_full, use_container_width=True)

            # Metrics table
            st.subheader("Performance Metrics")
            em_s21 = np.array(em_result.get("s21_mag", em_result.get("S21_mag", [])))
            em_s11 = np.array(em_result.get("s11_mag", em_result.get("S11_mag", [])))

            surr_metrics = compute_mse(y_star_st, selected_surr["s21_mag"],
                                       selected_surr["s11_mag"])
            em_metrics   = compute_mse(y_star_st, em_s21, em_s11) if len(em_s21) > 0 else {}

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Surrogate Prediction**")
                for k, v in surr_metrics.items():
                    if "Coverage" in k:
                        st.metric(k, f"{v*100:.0f}%")
                    else:
                        st.metric(k, f"{v:.4f}")
            with col2:
                st.markdown("**EM Ground Truth (OpenEMS)**")
                for k, v in em_metrics.items():
                    if "Coverage" in k:
                        st.metric(k, f"{v*100:.0f}%")
                    else:
                        st.metric(k, f"{v:.4f}")
            with col3:
                st.markdown("**Layout Properties**")
                st.metric("Fill fraction",
                          f"{selected_layout.mean()*100:.1f}%")
                st.metric("Connected",
                          "Yes" if is_connected(selected_layout) else "No")
                st.metric("Passivity OK",
                          "Yes" if em_result.get("passivity_ok", True) else "No")
                st.metric("KK residual",
                          f"{em_result.get('kk_residual', float('nan')):.3f}")

            # Surrogate-vs-EM accuracy
            if len(em_s21) == N_FREQ:
                surr_em_mse = float(np.mean(
                    (selected_surr["s21_mag"] - em_s21) ** 2 +
                    (selected_surr["s11_mag"] - em_s11) ** 2
                ) / 2)
                st.metric(
                    "Surrogate vs EM MSE",
                    f"{surr_em_mse:.4f}",
                    help="How accurately the surrogate predicted the real EM result",
                )

            # Download results
            import json
            result_bundle = {
                "spec_type":    filter_type,
                "substrate":    substrate_name,
                "method":       method,
                "T_steps":      T_steps,
                "layout":       selected_layout.tolist(),
                "surrogate_mse": surr_metrics,
                "em_mse":       em_metrics,
                "em_passivity_ok": bool(em_result.get("passivity_ok", True)),
            }
            st.download_button(
                "📥 Download Results (JSON)",
                data=json.dumps(result_bundle, indent=2),
                file_name="pixel_result.json",
                mime="application/json",
            )
        else:
            # Show just surrogate comparison if no EM yet
            st.subheader("Target vs Surrogate Prediction")
            st.caption("Click **Run OpenEMS Verification** above to add ground-truth EM results.")
            fig_surr = plot_sparams(
                y_star_st,
                surrogate=selected_surr,
                title="Surrogate Prediction vs Target",
            )
            st.plotly_chart(fig_surr, use_container_width=True)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
