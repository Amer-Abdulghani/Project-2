# module/E_x_E_y_run.py
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional


try:
    from module.jones import from_linear_terms as _from_linear_terms
except Exception as e:
    raise ImportError(f"Failed to import 'from_linear_terms' from module.jones: {e}. "
                     "Please ensure the jones module is available and properly implemented.")

# ----------------- helpers -----------------
def _np(a):
    return np.array(a) if a is not None else None

def lock_monitor_grid(fdtd, lam_start: float, lam_stop: float, npts: int, mon_name: str = "T") -> str:
    """Lock wavelength grid globally or per-monitor. Returns 'global' or 'per-monitor'."""
    try:
        fdtd.setglobalmonitor("wavelength start", float(lam_start))
        fdtd.setglobalmonitor("wavelength stop",  float(lam_stop))
        fdtd.setglobalmonitor("frequency points", int(npts))
        print("Locked grid using GLOBAL monitor settings.")
        return "global"
    except Exception:
        pass
    try:
        try: fdtd.setnamed(mon_name, "override global monitor settings", 1)
        except Exception: pass
        for k, v in [
            ("wavelength start", float(lam_start)),
            ("wavelength stop",  float(lam_stop)),
            ("frequency points", int(npts)),
            ("use linear wavelength spacing", 1),
        ]:
            try: fdtd.setnamed(mon_name, k, v)
            except Exception:
                if k == "use linear wavelength spacing":
                    try: fdtd.setnamed(mon_name, "use linear frequency spacing", 0)
                    except Exception: pass
        print(f"Locked grid on monitor '{mon_name}' (per-monitor).")
        return "per-monitor"
    except Exception as e:
        raise RuntimeError(f"Failed to lock grid: {e}")

def enable_field_output(fdtd, mon_name: str = "T") -> None:
    """Ensure complex E fields are recorded on the monitor."""
    for prop in [
        "record fields",
        "output E field",
        "save fields",
        "include field data",
        "calculate complex fields",
        "save complex fields",
        "output Ex",
        "output Ey",
    ]:
        try: fdtd.setnamed(mon_name, prop, 1)
        except Exception: pass

def _move_freq_last(F, f):
    """Move the axis whose length equals len(f) to the last position (if present)."""
    F = np.asarray(F)
    fN = np.asarray(f).size
    cand_f = [i for i, d in enumerate(F.shape) if d == fN]
    if cand_f:
        F = np.moveaxis(F, cand_f[0], -1)
    return F

def area_avg_weighted_xy(field, x, y, f) -> np.ndarray:
    """Area-weighted average over (x,y) using coordinate-aware trapezoids (integrate/A). Returns (Nf,)."""
    F = _move_freq_last(field, f)
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if F.ndim < 2:
        raise ValueError("Field array must have at least 2 dims for x and y.")
    # identify x,y axes among non-frequency dims
    ax_sizes = F.shape[:-1]
    axes = list(range(len(ax_sizes)))
    ax_x = next((i for i in axes if ax_sizes[i] == x.size), 0)
    ax_y = next((i for i in axes if ax_sizes[i] == y.size and i != ax_x), 1)
    F = np.moveaxis(F, (ax_x, ax_y), (0, 1))      # -> (Nx, Ny, ...others..., Nf)
    Fx  = np.trapezoid(F, x, axis=0)              # -> (Ny, ...others..., Nf)
    Fxy = np.trapezoid(Fx, y, axis=0)             # -> (...others..., Nf)
    A = (x[-1] - x[0]) * (y[-1] - y[0])
    if A == 0:
        raise ValueError("Area is zero — check monitor x/y spans.")
    Fxy = Fxy / A
    if Fxy.ndim > 1:                               # collapse extra non-freq dims (e.g., z=1)
        Fxy = Fxy.mean(axis=tuple(range(Fxy.ndim - 1)))
    return np.asarray(Fxy).reshape(-1)

def run_once(fdtd, pol_deg: float, mon_name: str = "T", src_name: str = "src",
             debug: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Set source polarization, run, and return (f, Ex0, Ey0) area-averaged complex amplitudes."""
    if fdtd.layoutmode() != 1:
        fdtd.switchtolayout()
    fdtd.setnamed(src_name, "polarization angle", float(pol_deg))
    fdtd.run()

    f  = _np(fdtd.getdata(mon_name, "f"))
    x  = _np(fdtd.getdata(mon_name, "x"))
    y  = _np(fdtd.getdata(mon_name, "y"))
    Ex = fdtd.getdata(mon_name, "Ex")
    Ey = fdtd.getdata(mon_name, "Ey")
    if f is None:
        raise RuntimeError(f"Monitor '{mon_name}' did not return frequency grid 'f'.")
    if any(v is None for v in (x, y, Ex, Ey)):
        raise RuntimeError(
            f"Missing x/y or fields on monitor '{mon_name}'. "
            f"Enable field recording (e.g., 'Record fields' / 'Output E field')."
        )
    if debug:
        print(f"[debug] Ex shape: {np.asarray(Ex).shape}, Ey shape: {np.asarray(Ey).shape}, "
              f"len(x)={x.size}, len(y)={y.size}, len(f)={f.size}")

    Ex0 = area_avg_weighted_xy(Ex, x, y, f)   # -> t_xx or t_xy
    Ey0 = area_avg_weighted_xy(Ey, x, y, f)   # -> t_yx or t_yy
    return f, Ex0, Ey0

# ----------------- main flows -----------------
def two_run_jones(fdtd,
                  lam_start: float, lam_stop: float, npts: int,
                  mon_name: str = "T", src_name: str = "src",
                  out_dir: Optional[Path] = None,
                  basename: str = "jones",
                  save_csv: bool = True,
                  debug: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full Ex/Ey pipeline:
      - lock grid, enable field output
      - run 0° and 90°; area-average Ex/Ey on monitor
      - assemble linear & circular Jones data using module.jones
      - optionally save CSVs to out_dir/basename_{linear|circular}.csv
    Returns: (df_lin, df_cir)
    """
    lock_monitor_grid(fdtd, lam_start, lam_stop, npts, mon_name=mon_name)
    enable_field_output(fdtd, mon_name=mon_name)

    f1, Ex_x0, Ey_x0 = run_once(fdtd, 0.0,  mon_name=mon_name, src_name=src_name, debug=debug)   # t_xx, t_yx
    f2, Ex_y0, Ey_y0 = run_once(fdtd, 90.0, mon_name=mon_name, src_name=src_name, debug=False)   # t_xy, t_yy

    if f1.shape != f2.shape or np.max(np.abs(f1 - f2)) > 1e-12:
        raise RuntimeError("Frequency grids differ between runs; ensure monitor grid is fixed.")

    c = 299_792_458.0
    lam_nm = (c / f1) * 1e9
    t_xx, t_yx = np.asarray(Ex_x0).reshape(-1), np.asarray(Ey_x0).reshape(-1)
    t_xy, t_yy = np.asarray(Ex_y0).reshape(-1), np.asarray(Ey_y0).reshape(-1)
    lam_nm     = np.asarray(lam_nm).reshape(-1)

    rows_lin, rows_cir = [], []
    for i in range(lam_nm.size):
        conv = _from_linear_terms(t_xx[i], t_xy[i], t_yx[i], t_yy[i])
        rows_lin.append({
            "lambda_nm": lam_nm[i],
            "txx": complex(t_xx[i]),
            "txy": complex(t_xy[i]),
            "tyx": complex(t_yx[i]),
            "tyy": complex(t_yy[i]),
        })
        Tc = conv["T_cir"]
        rows_cir.append({
            "lambda_nm": lam_nm[i],
            "T_LL": complex(Tc[0,0]),
            "T_LR": complex(Tc[0,1]),
            "T_RL": complex(Tc[1,0]),
            "T_RR": complex(Tc[1,1]),
        })

    df_lin = pd.DataFrame(rows_lin).sort_values("lambda_nm")
    df_cir = pd.DataFrame(rows_cir).sort_values("lambda_nm")

    if save_csv and out_dir is not None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{basename}_linear.csv").write_text(df_lin.to_csv(index=False))
        (out_dir / f"{basename}_circular.csv").write_text(df_cir.to_csv(index=False))
        print(f"Wrote:\n  {out_dir / (basename + '_linear.csv')}\n  {out_dir / (basename + '_circular.csv')}\nRows: {len(df_lin)}")

    # quick reciprocity heads-up (normal incidence, reciprocal media)
    try:
        diff = np.mean(np.abs(t_xy - t_yx))
        base = max(np.mean(np.abs(np.concatenate([t_xy, t_yx]))), 1e-12)
        print(f"[debug] reciprocity check: mean|t_xy - t_yx| / mean|t| ≈ {diff/base:.3e}")
    except Exception:
        pass

    return df_lin, df_cir

# ----------------- optional plotting -----------------
def plot_circular(df_cir: pd.DataFrame, use_mathtext: bool = True):
    """Return (fig, (ax_mag, ax_db)) plotting |T| and 20log10|T| with LaTeX-style labels."""
    import matplotlib.pyplot as plt
    lam = df_cir["lambda_nm"].to_numpy()
    T_LL = df_cir["T_LL"].to_numpy().astype(complex)
    T_LR = df_cir["T_LR"].to_numpy().astype(complex)
    T_RL = df_cir["T_RL"].to_numpy().astype(complex)
    T_RR = df_cir["T_RR"].to_numpy().astype(complex)

    mag = {k: np.abs(v) for k, v in dict(T_LL=T_LL, T_LR=T_LR, T_RL=T_RL, T_RR=T_RR).items()}
    eps = 1e-12
    db  = {k: 20*np.log10(np.maximum(m, eps)) for k, m in mag.items()}

    fig = plt.figure(figsize=(12,4.5))
    ax1 = plt.subplot(1,2,1); ax2 = plt.subplot(1,2,2)

    def L(s): return rf"${s}$" if use_mathtext else s  

    ax1.plot(lam, mag["T_RR"], label=L("T_{RR}"), linewidth=2)
    ax1.plot(lam, mag["T_LL"], label=L("T_{LL}"), linewidth=2)
    ax1.plot(lam, mag["T_LR"], label=L("T_{LR}"), linewidth=2)
    ax1.plot(lam, mag["T_RL"], label=L("T_{RL}"), linewidth=2)
    ax1.set_xlabel(L(r"\lambda (nm)")); ax1.set_ylabel(L(r" |T| "))
    ax1.set_ylim(0, 1.0); ax1.grid(True, alpha=0.3); ax1.legend(frameon=False)

    ax2.plot(lam, db["T_RR"], label=L("T_{RR}"), linewidth=2)
    ax2.plot(lam, db["T_LL"], label=L("T_{LL}"), linewidth=2)
    ax2.plot(lam, db["T_LR"], label=L("T_{LR}"), linewidth=2)
    ax2.plot(lam, db["T_RL"], label=L("T_{RL}"), linewidth=2)
    ax2.set_xlabel(L(r"\lambda (nm)")); ax2.set_ylabel(L(r" |T| (dB) "))
    ax2.set_ylim(-60, 1); ax2.grid(True, alpha=0.3); ax2.legend(frameon=False)

    fig.tight_layout()
    return fig, (ax1, ax2)
