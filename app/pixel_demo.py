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

# ── Demo presets: real S-param arrays drawn from held-out test samples ────────
# These ARE within the model's training distribution (11-primitive dataset).
# Using them as targets gives reliable, visually clean synthesis results.
DEMO_PRESETS: dict[str, dict] = {
    "Mid-band Notch  (9–14 GHz reject)": {
        "desc": "Coupled-line structure with S21 rejection 11–15 GHz, S11 well-matched at 9.8 GHz",
        "s11_mag": np.array([0.2027,0.1598,0.2626,0.4418,0.3213,0.3015,0.3902,0.3460,0.3737,0.3834,0.3881,0.4633,0.4322,0.4763,0.5815,0.5141,0.5509,0.5986,0.5333,0.5568,0.5472,0.5448,0.5965,0.5615,0.6026,0.6575,0.5922,0.6128,0.6051,0.5652,0.5919,0.5571,0.5550,0.5865,0.5326,0.5351,0.5360,0.4862,0.4889,0.4477,0.4053,0.4008,0.3346,0.2807,0.2416,0.1594,0.0909,0.0709,0.1360,0.2560,0.3523,0.4625,0.5914,0.6014,0.6511,0.7086,0.6769,0.7075,0.6971,0.6726,0.7127,0.6662,0.6546,0.6934,0.6372,0.6395,0.6607,0.6146,0.6356,0.6278,0.6029,0.6474,0.6050,0.6029,0.6681,0.5941,0.6036,0.6626,0.5858,0.6072,0.6134,0.5837,0.6592,0.5752,0.5292,0.5884,0.5076,0.5043,0.5424,0.4906,0.5462,0.5254,0.4752,0.5405,0.4406,0.3949,0.4457,0.3631,0.3505,0.4155], dtype=np.float32),
        "s21_mag": np.array([0.8156,0.6760,0.6538,0.6436,0.6172,0.5523,0.3633,0.5171,0.6389,0.4761,0.5572,0.6248,0.5148,0.5816,0.5783,0.5355,0.5908,0.4675,0.4860,0.5408,0.4013,0.4803,0.5337,0.4342,0.5307,0.5365,0.4716,0.5445,0.4690,0.4628,0.5326,0.4385,0.4882,0.5497,0.4586,0.5271,0.5446,0.4877,0.5743,0.5380,0.5212,0.6257,0.5591,0.5675,0.6667,0.5894,0.6127,0.6800,0.5922,0.6035,0.5932,0.4809,0.4600,0.3949,0.2819,0.2541,0.2084,0.1362,0.1406,0.1352,0.1223,0.1243,0.1171,0.1372,0.1291,0.1138,0.1390,0.1315,0.1252,0.1384,0.1394,0.1563,0.1498,0.1663,0.2115,0.1802,0.2148,0.2751,0.2247,0.2795,0.3190,0.2827,0.3935,0.3639,0.3097,0.4438,0.3883,0.3570,0.5017,0.4466,0.4661,0.5760,0.5274,0.5789,0.5684,0.5428,0.6340,0.5911,0.6028,0.7411], dtype=np.float32),
        "s11_phase": np.array([-1.1982,-1.9599,-2.2289,-2.5992,-2.3350,-0.8560,0.4072,-0.9443,-0.7084,0.4331,-1.0072,-0.8998,0.4768,0.1265,1.5578,1.6010,0.1446,1.3970,1.4788,-0.0102,1.4318,2.7410,2.7827,2.6830,2.7055,2.5038,2.5068,2.5369,2.3332,2.3437,2.3046,2.1621,2.1615,2.0384,1.9706,1.9650,1.8013,1.7694,1.7331,1.5718,1.5412,1.4486,1.3130,1.2713,1.1312,1.1062,1.4030,0.8557,-1.4758,-1.3778,1.3852,2.6686,2.5842,2.3600,2.2293,1.9917,1.8674,1.7753,1.5724,1.5062,1.4096,1.2557,1.2142,1.0895,0.9871,0.9531,0.7988,0.7411,0.7011,0.5311,0.5043,0.4406,0.2794,0.2663,0.1530,0.0360,0.0247,-0.1632,-0.1982,-0.1989,-0.4510,-0.4059,-0.4040,-0.6669,-0.6316,-0.7498,-0.8863,-0.9093,-1.3075,-1.1208,-0.9299,-1.4080,-1.0271,-0.6462,-1.2137,-0.5360,0.2701,-0.6860,-0.0697,1.7295], dtype=np.float32),
        "s21_phase": np.array([-0.1482,-0.2135,-0.7639,-1.4785,-0.8759,-0.6682,-0.9859,-0.7137,-0.5543,-0.5813,-0.7026,-0.8923,-0.8364,-1.1642,-1.7063,-1.3035,-1.3937,-1.8917,-1.5139,-1.5162,-1.7275,-1.6368,-1.8322,-1.7313,-1.7922,-2.1536,-1.9233,-2.0070,-2.3370,-2.1285,-2.2661,-2.4535,-2.3276,-2.5344,-2.4286,-1.0916,0.3562,-1.1541,-1.1977,0.1212,-1.3798,-1.5049,-0.0162,-0.2434,1.3812,2.4926,2.6170,2.3174,2.1225,2.0521,1.6747,1.4629,1.3748,1.0068,0.8239,0.8328,0.3567,0.0422,0.3786,-0.3773,-1.1626,-0.1458,0.5675,1.3991,0.7929,0.1222,0.8480,0.2780,-0.4354,0.3330,-0.2602,-0.9414,-0.1422,-0.8517,-1.4806,-0.4610,0.0147,1.0767,0.5065,-0.3564,0.5055,-0.0266,-0.8684,0.0686,0.7077,1.5410,0.8931,0.0722,0.5906,0.2772,-0.3655,0.1156,-0.0239,-0.4680,-0.0665,0.0433,0.0593,0.0692,-0.0315,0.1339], dtype=np.float32),
    },
    "X-band Notch  (7–10 GHz reject)": {
        "desc": "Resonator structure with S21 rejection at 7.8–9.2 GHz, deep S11 match at 13.5 GHz",
        "s11_mag": np.array([0.2901,0.1927,0.2944,0.4419,0.2987,0.2942,0.3409,0.2870,0.3286,0.3090,0.3192,0.4066,0.3524,0.4100,0.5252,0.4286,0.4757,0.5506,0.4646,0.4948,0.4982,0.4891,0.5579,0.5302,0.5750,0.6492,0.5823,0.6073,0.6210,0.5902,0.6369,0.6133,0.6138,0.6688,0.6265,0.6270,0.6509,0.6266,0.6470,0.6487,0.6405,0.6745,0.6558,0.6642,0.7065,0.6650,0.6839,0.7102,0.6640,0.6907,0.6534,0.6035,0.6167,0.5473,0.5267,0.5266,0.4382,0.4216,0.3940,0.3073,0.2792,0.2339,0.1756,0.1555,0.1079,0.0610,0.0558,0.0592,0.0690,0.0732,0.1134,0.1533,0.1566,0.1923,0.2192,0.2221,0.2714,0.2781,0.2984,0.3906,0.3469,0.3225,0.3807,0.3391,0.3406,0.3684,0.3378,0.3951,0.3826,0.3726,0.5005,0.4161,0.3367,0.4087,0.3359,0.3063,0.3574,0.3034,0.3319,0.4025], dtype=np.float32),
        "s21_mag": np.array([0.8547,0.7470,0.7222,0.6922,0.6908,0.6349,0.4371,0.5845,0.6848,0.5465,0.6375,0.6822,0.5955,0.6755,0.6456,0.6039,0.6460,0.5132,0.5505,0.5909,0.4434,0.5411,0.5756,0.4748,0.5822,0.5526,0.4764,0.5413,0.4362,0.4204,0.4744,0.3500,0.3638,0.3898,0.2723,0.2846,0.2672,0.1757,0.1970,0.1734,0.1146,0.1303,0.1377,0.1381,0.1386,0.1973,0.2488,0.2410,0.3374,0.4230,0.3929,0.4861,0.5599,0.5243,0.6350,0.6770,0.6355,0.7489,0.7331,0.6891,0.7670,0.7038,0.7160,0.7856,0.6971,0.7613,0.8094,0.7132,0.8387,0.8409,0.6971,0.7918,0.7358,0.6409,0.7373,0.6567,0.6405,0.7198,0.6465,0.7588,0.7525,0.5768,0.6933,0.6429,0.4719,0.6236,0.5694,0.4872,0.6341,0.6115,0.6868,0.7236,0.5961,0.6758,0.6443,0.5461,0.6689,0.6367,0.6118,0.7815], dtype=np.float32),
        "s11_phase": np.array([-0.4865,-0.6888,-1.2248,-2.0018,-1.5531,-0.2605,0.7284,-0.4573,-0.3404,0.6171,-0.6788,-0.6442,0.5780,0.2494,1.5344,1.5482,0.1346,1.2506,1.3210,-0.1690,1.1501,2.4277,2.3985,2.2213,2.2349,1.9528,1.8732,1.8743,1.6014,1.5217,1.4294,1.2209,1.1393,0.9722,0.8160,0.7251,0.5279,0.3936,0.2762,0.0718,-0.0537,-0.2039,-0.4054,-0.5298,-0.7218,-0.9013,-1.0351,-1.2829,-1.4140,-1.5608,-1.8423,-1.9392,-2.1273,-2.3672,-2.4608,-2.7264,-2.7274,-1.3706,1.4579,2.8256,2.8655,2.5843,2.5627,2.3881,2.1591,2.2538,1.6614,0.2460,-0.9836,-1.4567,-1.3655,-1.5359,-1.7296,-1.7064,-1.9360,-2.0045,-2.0215,-2.3640,-2.1230,-0.8171,0.4896,-0.8884,-0.9188,0.4307,0.1444,1.4041,1.4986,0.1422,1.1400,1.4648,0.0929,-0.2276,-1.2337,-0.5664,0.0494,-0.8058,0.0264,0.3920,-0.5512,-0.0319], dtype=np.float32),
        "s21_phase": np.array([-0.1210,-0.1846,-0.7198,-1.4167,-0.8059,-0.4650,-0.6090,-0.4997,-0.4568,-0.5075,-0.5977,-0.7414,-0.7364,-1.1108,-1.6858,-1.2600,-1.3276,-1.8418,-1.4658,-1.4404,-1.6282,-1.6107,-1.8632,-1.7191,-1.8196,-2.2285,-1.9717,-2.1050,-2.4630,-2.2328,-2.4384,-2.6508,-2.4787,-2.7351,-2.6611,-1.2706,0.1551,-1.3165,-1.3046,0.3315,-0.4210,-1.1349,-0.1783,-0.5651,-1.2172,-0.7402,-0.7956,-1.0595,-0.9440,-1.1020,-1.2760,-1.2824,-1.4999,-1.5926,-1.6663,-1.9530,-1.9402,-2.0328,-2.3301,-2.2636,-2.3906,-2.6251,-2.4175,-1.3100,0.3873,0.1878,1.6354,1.6046,0.0876,1.3665,1.4256,-0.2831,-0.1539,-0.4279,1.1120,0.9211,-0.4517,0.7438,0.6831,-0.5889,0.4841,0.4880,-0.7888,0.4910,1.7278,1.5007,0.3603,-0.7431,0.0398,0.0690,-0.6849,-0.0294,0.1895,-0.1052,0.1033,0.1250,0.1891,0.1031,-0.1539,0.0691], dtype=np.float32),
    },
    "Matched Resonator  (S11 −34 dB @ 10 GHz)": {
        "desc": "Stub resonator with exceptionally deep S11 match (−33.8 dB) at 9.8 GHz",
        "s11_mag": np.array([0.1748,0.1182,0.2323,0.4092,0.2652,0.2186,0.2677,0.2417,0.2777,0.2731,0.2869,0.3630,0.3221,0.3702,0.4795,0.3964,0.4115,0.4538,0.3975,0.4201,0.4117,0.4070,0.4558,0.4203,0.4498,0.5053,0.4372,0.4414,0.4454,0.4025,0.4137,0.3903,0.3768,0.3931,0.3517,0.3478,0.3504,0.3011,0.2891,0.2604,0.2174,0.2017,0.1642,0.1276,0.1050,0.0648,0.0290,0.0204,0.0465,0.0883,0.1167,0.1534,0.2012,0.2179,0.2532,0.2953,0.3072,0.3549,0.3843,0.3966,0.4588,0.4496,0.4511,0.5020,0.4796,0.4989,0.5293,0.5174,0.5777,0.5741,0.5720,0.6619,0.5969,0.5812,0.6468,0.5877,0.6210,0.6578,0.6394,0.7481,0.6856,0.6448,0.7490,0.6411,0.6399,0.7320,0.6372,0.6938,0.7153,0.6067,0.6413,0.5669,0.5256,0.5899,0.4964,0.4961,0.5456,0.4579,0.4748,0.4884], dtype=np.float32),
        "s21_mag": np.array([0.8309,0.7142,0.6920,0.6789,0.6664,0.6193,0.4620,0.5976,0.7059,0.5668,0.6413,0.7039,0.6080,0.6732,0.6752,0.6360,0.6772,0.5519,0.5877,0.6508,0.5230,0.6070,0.6628,0.5719,0.6769,0.6883,0.6189,0.6863,0.6117,0.6120,0.6800,0.5861,0.6488,0.7089,0.6194,0.7119,0.7319,0.6640,0.7563,0.6980,0.6695,0.7612,0.6768,0.6992,0.7810,0.6817,0.7318,0.7722,0.6856,0.7619,0.7344,0.6722,0.7694,0.6930,0.6518,0.7357,0.6500,0.6606,0.7205,0.6345,0.6904,0.6791,0.5840,0.6587,0.6093,0.5384,0.6251,0.5655,0.5379,0.6187,0.5538,0.5620,0.5861,0.5102,0.5411,0.5291,0.4662,0.5311,0.4853,0.4051,0.4889,0.4106,0.2682,0.3972,0.4385,0.3765,0.4534,0.5221,0.4921,0.4980,0.5236,0.4905,0.5182,0.5489,0.5240,0.5907,0.6220,0.6030,0.6908,0.6692], dtype=np.float32),
        "s11_phase": np.array([-0.9262,-1.4839,-1.7268,-2.2255,-1.9330,-0.4687,0.7386,-0.7066,-1.7943,-2.3683,-2.2050,-2.1283,-2.4448,-2.4692,-2.7246,-2.5658,-1.1554,0.2269,-1.2395,-1.1529,0.1408,-1.3815,-1.4484,0.1254,-0.1065,1.3982,1.3987,-0.1148,1.3706,2.6908,2.8126,2.6516,2.6676,2.5607,2.4797,2.4974,2.3351,2.3072,2.3063,2.1393,2.1292,2.0800,1.9427,1.9417,1.8334,1.6405,0.7137,-1.2717,-1.6968,-1.6293,-1.7760,-1.7695,-1.8522,-1.9642,-1.9647,-2.0951,-2.1441,-2.1473,-2.3120,-2.3063,-2.3247,-2.4802,-2.4554,-2.5301,-2.6195,-2.5924,-2.7439,-2.6009,-1.2659,0.2425,-1.2563,-1.2486,0.1588,-1.3539,-1.4655,0.2084,0.0061,1.4665,1.5816,-0.0105,-0.0197,-1.5923,-1.5707,-0.0700,-0.3579,1.0095,1.1828,-0.1807,0.9573,1.2871,0.0405,1.1654,1.8852,0.9881,0.3681,-0.7543,0.2097,0.7135,-0.3736,0.2074], dtype=np.float32),
        "s21_phase": np.array([-0.1197,-0.1760,-0.6665,-1.3069,-0.7480,-0.4223,-0.5337,-0.4624,-0.4594,-0.5166,-0.6029,-0.7602,-0.7272,-0.9766,-1.4173,-1.0997,-1.1313,-1.4993,-1.2382,-1.2340,-1.3834,-1.3544,-1.5434,-1.4826,-1.5395,-1.8524,-1.6727,-1.7255,-2.0045,-1.8482,-1.9507,-2.1041,-2.0276,-2.2161,-2.2297,-2.2151,-2.4520,-2.3872,-2.4304,-2.6535,-2.5727,-2.6843,-2.8456,-2.6430,-1.5192,0.2287,0.0037,1.4895,1.4704,-0.0944,1.3856,2.7088,2.7738,2.5863,2.6460,2.4871,2.3575,2.4414,2.2080,2.1397,2.2170,1.9616,1.9365,1.9392,1.7322,1.7611,1.6297,1.5086,1.6249,1.3661,1.2779,1.4369,1.1160,0.9842,1.0701,0.7998,0.6583,0.6456,0.4488,0.3355,0.2639,-0.3694,-1.0803,-0.4392,0.2027,0.6196,0.1977,-0.0954,0.1496,-0.1401,-0.3919,-0.0979,-0.0243,0.1368,0.0498,-0.0391,0.2447,0.0312,-0.1242,0.2473], dtype=np.float32),
    },
    "High-freq Stub  (S11 −29 dB @ 12 GHz)": {
        "desc": "Shunt stub with sharp S11 match at 12.3 GHz, S21 notch at 15–16 GHz",
        "s11_mag": np.array([0.2107,0.1457,0.2514,0.4292,0.2984,0.2772,0.3524,0.3089,0.3418,0.3455,0.3525,0.4310,0.3953,0.4426,0.5510,0.4751,0.5087,0.5628,0.4956,0.5187,0.5131,0.5096,0.5629,0.5297,0.5705,0.6287,0.5604,0.5790,0.5816,0.5410,0.5674,0.5409,0.5400,0.5757,0.5298,0.5406,0.5540,0.5111,0.5268,0.5006,0.4746,0.4984,0.4533,0.4342,0.4460,0.4014,0.3894,0.3804,0.3444,0.3386,0.3108,0.2809,0.2742,0.2349,0.2049,0.1869,0.1455,0.1116,0.0792,0.0456,0.0346,0.0561,0.1000,0.1568,0.1856,0.2351,0.2933,0.3115,0.3629,0.3935,0.4129,0.4759,0.4676,0.4930,0.5684,0.5233,0.5525,0.6226,0.5643,0.5979,0.6142,0.5986,0.6838,0.5987,0.5684,0.6411,0.5483,0.5571,0.5995,0.5406,0.5903,0.4989,0.4273,0.4529,0.2743,0.2069,0.2927,0.2925,0.3286,0.3931], dtype=np.float32),
        "s21_mag": np.array([0.8429,0.7079,0.6813,0.6643,0.6478,0.5896,0.4030,0.5517,0.6658,0.5131,0.5959,0.6567,0.5549,0.6252,0.6156,0.5738,0.6237,0.4982,0.5248,0.5781,0.4405,0.5250,0.5742,0.4785,0.5800,0.5777,0.5115,0.5811,0.5007,0.4985,0.5624,0.4644,0.5188,0.5711,0.4800,0.5523,0.5576,0.5022,0.5847,0.5303,0.5186,0.6094,0.5259,0.5419,0.6115,0.5315,0.5758,0.6067,0.5500,0.6261,0.6131,0.5842,0.6864,0.6338,0.6278,0.7343,0.6600,0.6746,0.7486,0.6809,0.7278,0.7366,0.6794,0.7523,0.6985,0.6257,0.6756,0.6084,0.5314,0.5438,0.4880,0.4152,0.4156,0.3686,0.2860,0.3071,0.2748,0.1762,0.2445,0.2994,0.2633,0.3060,0.3780,0.3409,0.3559,0.4298,0.3978,0.4324,0.5218,0.4997,0.5526,0.5758,0.5615,0.5961,0.5431,0.6127,0.6963,0.5802,0.6495,0.8235], dtype=np.float32),
        "s11_phase": np.array([-1.0441,-1.7072,-1.9521,-2.4249,-2.1594,-0.6875,0.5193,-0.8123,-0.5909,0.5187,-0.8988,-0.8083,0.5482,0.2129,1.6246,1.6555,0.0698,0.0022,-1.6131,-1.4014,1.3155,2.7638,2.8037,2.6875,2.7142,2.5013,2.4954,2.5306,2.3136,2.3205,2.2846,2.1301,2.1305,2.0082,1.9328,1.9335,1.7702,1.7370,1.7126,1.5559,1.5322,1.4581,1.3364,1.3107,1.1876,1.1041,1.0685,0.9161,0.8560,0.7959,0.6385,0.5840,0.4833,0.3443,0.2841,0.1400,0.0436,0.0012,-0.0544,0.3579,1.2276,1.7294,1.9269,1.8528,1.6823,1.5911,1.3580,1.2186,1.1124,0.8557,0.7525,0.6239,0.3963,0.3176,0.1487,-0.0171,-0.0818,-0.3139,-0.3841,-0.4278,-0.7221,-0.6961,-0.7209,-1.0317,-1.0364,-1.2052,-1.3687,-1.4750,-2.0046,-1.6553,-0.2163,0.8251,-0.4345,-0.2325,0.9291,0.3314,-1.3069,-1.1556,0.3494,2.1177], dtype=np.float32),
        "s21_phase": np.array([-0.1369,-0.2000,-0.7333,-1.4271,-0.8311,-0.5402,-0.7370,-0.5842,-0.5182,-0.5649,-0.6702,-0.8450,-0.8034,-1.1152,-1.6358,-1.2524,-1.3214,-1.7856,-1.4398,-1.4316,-1.6153,-1.5568,-1.7614,-1.6614,-1.7258,-2.0791,-1.8531,-1.9296,-2.2424,-2.0422,-2.1692,-2.3306,-2.2171,-2.4167,-2.4125,-2.3838,-2.6214,-2.3940,-1.1457,0.3356,-1.1179,-1.2350,0.3126,0.1347,1.6612,1.5876,0.0733,1.4350,1.4196,-0.1500,1.3433,2.6602,2.7152,2.5308,2.5638,2.4016,2.2518,2.2803,2.0474,1.9300,1.9287,1.6641,1.5559,1.4869,1.2442,1.1438,1.0085,0.8057,0.7208,0.5736,0.3830,0.2965,0.1886,-0.0222,-0.1244,-0.1108,0.1016,0.6058,0.0784,-0.1995,0.3638,-0.2259,-0.7385,-0.1485,-0.7745,-1.5405,-0.6116,-0.0313,0.4985,0.1612,-0.2763,0.0501,-0.0236,-0.1918,-0.1318,-0.0392,0.3534,0.1788,-0.0728,0.2561], dtype=np.float32),
    },
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
        return simulate(layout, meta={"type": 0}, substrate_id=substrate_id)
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
            fig.add_hline(y=lvl, line=dict(color="rgba(150,150,150,0.35)",
                                            dash="dot", width=1),
                          annotation_text=lbl,
                          annotation_font=dict(size=8, color="#aaa"),
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
        fig.update_yaxes(title_text="Magnitude (dB)", range=[-25, 2],
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
    # SIDEBAR — spec selection + generation settings
    # ════════════════════════════════════════════════════════════════════════
    with st.sidebar:
        st.header("Target Specification")

        spec_source = st.radio(
            "Spec source",
            ["Demo Presets  ✓", "Custom (analytical)"],
            index=0,
            help="Demo Presets are real responses from the training distribution — "
                 "they reliably produce good synthesis results.",
        )

        if spec_source == "Demo Presets  ✓":
            preset_name = st.selectbox(
                "Select preset",
                list(DEMO_PRESETS.keys()),
                index=0,
            )
            p   = DEMO_PRESETS[preset_name]
            st.caption(p["desc"])
            y   = np.stack([p["s11_mag"], p["s21_mag"],
                            p["s11_phase"] / np.pi, p["s21_phase"] / np.pi])
            slbl = preset_name.split("  ")[0]   # short name

        else:
            # Advanced analytical specs — may be outside training distribution
            st.caption(
                "⚠️ Analytical specs (Bandpass, etc.) may be outside the "
                "model's training distribution — results may not match the target."
            )
            ftype = st.radio(
                "Filter type",
                ["Wideband", "Lowpass", "Bandstop", "Bandpass"],
                index=0,
            )
            if ftype == "Bandpass":
                fc  = st.slider("Centre frequency (GHz)", 1.0, 18.0, 6.0, 0.5)
                bw  = st.slider("3 dB bandwidth (GHz)", 0.5, 8.0, 2.0, 0.5)
                y   = spec_bandpass(fc, bw)
                slbl = f"Bandpass  {fc:.1f} GHz"
            elif ftype == "Bandstop":
                fc  = st.slider("Centre frequency (GHz)", 1.0, 18.0, 8.0, 0.5)
                bw  = st.slider("Rejection BW (GHz)", 0.5, 6.0, 1.5, 0.5)
                rej = st.slider("Stopband rejection (dB)", 10, 30, 15, 5)
                y   = spec_bandstop(fc, bw, rej)
                slbl = f"Bandstop  {fc:.1f} GHz / {rej} dB"
            elif ftype == "Lowpass":
                fc  = st.slider("Cutoff frequency (GHz)", 3.0, 12.0, 6.0, 0.5)
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
        k_cands  = st.slider("K candidates", 1, 5, 3)
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
