"""
Generate UACR-format reference channels and Python reference outputs, used as
committed fixtures by test/test_replay.jl (so the suite never calls Python).

Reference: uwa-channels/python `replay.py` (post-2026-07-07: no output
normalisation, 2*real() upconversion, spline interpolation with zero fill).
Place that file alongside this script as `replay_ref.py`.

Run from the repo root:  python test/data/gen_references.py

IMPORTANT - parameter constraint:
  The reference downconverts a REAL input, so its baseband retains an image at
  -2*fc. That image is only removed by the resample to fs_delay if
      2*fc > fs_delay/2.
  Configurations violating this leave the image in place and the output comes
  out 2x too large. The parameters below satisfy it (fc=12k, fs_delay=24k).
  The probe is passed as a REAL signal (the reference's 2*real() upconversion
  already compensates for the halving, matching Julia's analytic front end).

Needs: numpy, scipy, h5py, hdf5storage.
"""
import os
import numpy as np
import h5py
import hdf5storage
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "replay_ref", os.path.join(os.path.dirname(__file__), "replay_ref.py"))
ref = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ref)

OUTDIR = os.path.dirname(__file__)

# must mirror the constants in test_replay.jl
FS_IN, FC, FS_DELAY = 96_000.0, 12_000.0, 24_000.0
SEED = 20260812


def chirp(fs, D=0.008, f0=9_000.0, f1=15_000.0, tau=0.002):
    n = np.arange(int(round(D * fs))); t = n / fs
    k = (f1 - f0) / D
    x = np.cos(2 * np.pi * (f0 * t + 0.5 * k * t * t))
    w = np.ones_like(t)
    w[t < tau] = 0.5 * (1 - np.cos(np.pi * t[t < tau] / tau))
    w[t > D - tau] = 0.5 * (1 - np.cos(np.pi * (D - t[t > D - tau]) / tau))
    return x * w


def build(mode, L=16, M=2, T=480, step=1, time_varying=False):
    """UACR channel dict in MATLAB layout: h_hat is [delay, rx, time]."""
    rng = np.random.default_rng(SEED)
    fs_time = FS_DELAY / step
    h = (rng.standard_normal((L, M, T)) + 1j * rng.standard_normal((L, M, T))) * 0.15
    if time_varying:                      # smooth drift across snapshots
        h *= (1 + 0.3 * np.cos(2 * np.pi * np.arange(T) / T))[None, None, :]
    mat = {
        "version": np.array([[1.0]]),
        "h_hat": h.astype(np.complex128),
        "params": {"fs_delay": np.array([[FS_DELAY]]),
                   "fs_time": np.array([[fs_time]]),
                   "fc": np.array([[FC]])},
    }
    nphase = T * step                     # spec: phase spans the IR duration
    if mode == "phi":
        mat["phi_hat"] = np.cumsum(rng.standard_normal((M, nphase)) * 0.01, axis=1)
    elif mode == "theta":
        mat["theta_hat"] = np.cumsum(rng.standard_normal((M, nphase)) * 0.01, axis=1)
    return mat


def py_channel(path):
    """Read back the written fixture the way the Python reference expects."""
    with h5py.File(path, "r") as f:
        hh = f["h_hat"][:]                                  # [time, rx, delay]
        ch = {"version": f["version"][()],
              "h_hat": {"real": np.array(hh["real"]), "imag": np.array(hh["imag"])},
              "params": {k: f["params"][k][()] for k in f["params"]}}
        for k in ("phi_hat", "theta_hat"):
            if k in f:
                ch[k] = f[k][:]                             # [time, rx]
    return ch


CASES = [("none",  dict(step=1,  T=240, time_varying=False)),
         ("theta", dict(step=1,  T=240, time_varying=False)),
         ("phi",   dict(step=1,  T=240, time_varying=False)),
         ("tv",    dict(step=20, T=120, time_varying=True))]

for name, kw in CASES:
    mode = "phi" if name == "tv" else name
    mat = build(mode, **kw)
    tmp = os.path.join(OUTDIR, f"_tmp_{name}.mat")
    final = os.path.join(OUTDIR, f"replay_ref_{name}.mat")
    for p in (tmp, final):
        if os.path.exists(p):
            os.remove(p)

    hdf5storage.savemat(tmp, mat, format="7.3", matlab_compatible=True)
    probe = chirp(FS_IN)                       # REAL probe (see note above)
    y_ref = ref.replay(probe, FS_IN, np.arange(mat["h_hat"].shape[1]),
                       py_channel(tmp), start=0)
    os.remove(tmp)

    mat["probe"] = probe.astype(np.float64)
    mat["y_ref"] = np.asarray(y_ref, dtype=np.float64)
    hdf5storage.savemat(final, mat, format="7.3", matlab_compatible=True)
    print(f"{name:6s} step={kw['step']:2d}  y_ref{y_ref.shape}  "
          f"peak={np.abs(y_ref).max():.4f}  finite={np.isfinite(y_ref).all()}")
