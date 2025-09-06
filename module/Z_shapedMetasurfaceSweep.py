# module/Z_shapedMetasurfaceSweep.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple, Optional, List, Sequence, Dict
import numpy as np
import pandas as pd
import importlib

# Reuse your existing two-pass pipeline (direct, in-session runs)
import module.E_x_E_y_run as exy
exy = importlib.reload(exy)

# ---------------- config ----------------
@dataclass(frozen=True)
class ZSweepConfig:
    # wavelength band (nm) + sampling
    lam_start_nm: float
    lam_stop_nm: float
    npts: int

    # outputs (master-only)
    out_dir: Path
    master_prefix: str = "Z_sweep"   # -> Z_sweep_linear_ALL.csv, Z_sweep_circular_ALL.csv
    save_master: bool = True

    # solver controls (optional; None => leave solver as-is)
    sim_time_ps: Optional[float] = None
    auto_shutoff_level: Optional[float] = None

    # object names in the model tree
    mon_name: str = "T"
    src_name: str = "src"
    sg_name: str  = "Z"              # structure group to update

    # structure constants (nm) that don't change in the sweep
    t_nm: float = 50.0
    t_z_nm: float = 50.0
    mat_z: str = "Ag"

# ---------------- helpers ----------------
def _nm_to_m(x_nm: float) -> float:
    return float(x_nm) * 1e-9

def _set_solver(fdtd, sim_time_ps: Optional[float], shutoff: Optional[float]):
    if sim_time_ps is not None:
        try: fdtd.setnamed("FDTD", "simulation time", sim_time_ps * 1e-12)
        except Exception: pass
    if shutoff is not None:
        try: fdtd.setnamed("FDTD", "auto shutoff level", float(shutoff))
        except Exception: pass

def _update_Z_script(fdtd, sg_name: str,
                     l1_nm: float, l2_nm: float, l3_nm: float,
                     t_nm: float, t_z_nm: float, mat_z: str) -> None:
    """Update the 'Z' structure-group script with new (l1,l2,l3) (internally meters)."""
    l1 = _nm_to_m(l1_nm); l2 = _nm_to_m(l2_nm); l3 = _nm_to_m(l3_nm)
    t  = _nm_to_m(t_nm);  t_z = _nm_to_m(t_z_nm)
    sg_script = f'''
# ===== user parameters (meters) =====
l1 = {l1};
l3 = {l3};
l2 = {l2};
t  = {t};
t_z = {t_z};
mat = "{mat_z}";

vtx = [
  t,        l2/2;
  l1,       l2/2;
  l1,       l2/2 + t;
  0,        l2/2 + t;
  0,       -l2/2;
  t - l3,  -l2/2;
  t - l3,  -l2/2 - t;
  t,       -l2/2 - t
];

addpoly;
set("name","Ag");
set("material", mat);
set("vertices", vtx);
set("z span", t_z);
set("x", 0);
set("y", 0);
set("z", t_z/2);
'''
    fdtd.setnamed(sg_name, "script", sg_script)

# ---------------- direct, sequential sweep (existing) ----------------
def sweep_grid(fdtd,
               l1_list_nm: Iterable[float],
               l2_list_nm: Iterable[float],
               l3_list_nm: Iterable[float],
               cfg: ZSweepConfig,
               progress: bool = True
               ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cartesian sweep over l1,l2,l3 (nm). Returns (df_linear_ALL, df_circular_ALL)."""
    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    _set_solver(fdtd, cfg.sim_time_ps, cfg.auto_shutoff_level)

    lam_start_m = _nm_to_m(cfg.lam_start_nm)
    lam_stop_m  = _nm_to_m(cfg.lam_stop_nm)

    l1_vals = list(l1_list_nm); l2_vals = list(l2_list_nm); l3_vals = list(l3_list_nm)
    total = len(l1_vals) * len(l2_vals) * len(l3_vals)

    L_all: List[pd.DataFrame] = []
    C_all: List[pd.DataFrame] = []

    k = 0
    for l1 in l1_vals:
        for l2 in l2_vals:
            for l3 in l3_vals:
                k += 1
                if progress:
                    print(f"[Z-sweep] ({k}/{total}) l1={l1} nm, l2={l2} nm, l3={l3} nm")

                if fdtd.layoutmode() != 1:
                    fdtd.switchtolayout()
                _update_Z_script(fdtd, cfg.sg_name, l1, l2, l3, cfg.t_nm, cfg.t_z_nm, cfg.mat_z)

                df_lin, df_cir = exy.two_run_jones(
                    fdtd,
                    lam_start=lam_start_m, lam_stop=lam_stop_m, npts=cfg.npts,
                    mon_name=cfg.mon_name, src_name=cfg.src_name,
                    out_dir=None, basename="", save_csv=False, debug=False
                )

                for df in (df_lin, df_cir):
                    df["l1_nm"] = float(l1)
                    df["l2_nm"] = float(l2)
                    df["l3_nm"] = float(l3)

                L_all.append(df_lin)
                C_all.append(df_cir)

    dfL = pd.concat(L_all, ignore_index=True)
    dfC = pd.concat(C_all, ignore_index=True)

    if cfg.save_master:
        lin_path = out_dir / f"{cfg.master_prefix}_linear_ALL.csv"
        cir_path = out_dir / f"{cfg.master_prefix}_circular_ALL.csv"
        lin_path.write_text(dfL.to_csv(index=False))
        cir_path.write_text(dfC.to_csv(index=False))
        print(f"[Z-sweep] wrote:\n  {lin_path}\n  {cir_path}")

    return dfL, dfC

# ---------------- Job-Manager (parallel) version ----------------
def _save_jobfile(fdtd, filepath: Path, pol_deg: float, cfg: ZSweepConfig) -> None:
    """
    Save an .fsp that will run a single pass (given polarization).
    Assumes geometry & wavelength grid already configured in the model.
    """
    if fdtd.layoutmode() != 1:
        fdtd.switchtolayout()
    try:
        fdtd.setnamed(cfg.src_name, "polarization angle", float(pol_deg))
    except Exception:
        raise RuntimeError(f"Source '{cfg.src_name}' not found or can't set polarization.")
    # ensure wavelength grid & solver settings are persisted
    _set_solver(fdtd, cfg.sim_time_ps, cfg.auto_shutoff_level)
    # Lock the monitor grid (use existing helper)
    exy.lock_monitor_grid(fdtd, _nm_to_m(cfg.lam_start_nm), _nm_to_m(cfg.lam_stop_nm), cfg.npts, mon_name=cfg.mon_name)
    exy.enable_field_output(fdtd, mon_name=cfg.mon_name)
    fdtd.save(str(filepath))

def _read_fields_from_loaded(fdtd, mon_name: str) -> Dict[str, np.ndarray]:
    """Read f,x,y,Ex,Ey from the *currently loaded* file."""
    f  = exy._np(fdtd.getdata(mon_name, "f"))
    x  = exy._np(fdtd.getdata(mon_name, "x"))
    y  = exy._np(fdtd.getdata(mon_name, "y"))
    Ex = fdtd.getdata(mon_name, "Ex")
    Ey = fdtd.getdata(mon_name, "Ey")
    if f is None or x is None or y is None or Ex is None or Ey is None:
        raise RuntimeError(f"Missing T-monitor data in '{mon_name}'.")
    return {"f": f, "x": x, "y": y, "Ex": Ex, "Ey": Ey}

def _postprocess_two_files(fdtd, file_pol0: Path, file_pol90: Path, cfg: ZSweepConfig
                           ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load two result files (pol=0°,90°), compute area-averaged terms and build
    (df_linear, df_circular) for that geometry.
    """
    # --- pol 0 ---
    fdtd.load(str(file_pol0))
    D0 = _read_fields_from_loaded(fdtd, cfg.mon_name)
    Ex_x0 = exy.area_avg_weighted_xy(D0["Ex"], D0["x"], D0["y"], D0["f"])  # -> t_xx
    Ey_x0 = exy.area_avg_weighted_xy(D0["Ey"], D0["x"], D0["y"], D0["f"])  # -> t_yx
    # --- pol 90 ---
    fdtd.load(str(file_pol90))
    D90 = _read_fields_from_loaded(fdtd, cfg.mon_name)
    Ex_y0 = exy.area_avg_weighted_xy(D90["Ex"], D90["x"], D90["y"], D90["f"])  # -> t_xy
    Ey_y0 = exy.area_avg_weighted_xy(D90["Ey"], D90["x"], D90["y"], D90["f"])  # -> t_yy

    # frequency grids should match
    if D0["f"].shape != D90["f"].shape or np.max(np.abs(D0["f"] - D90["f"])) > 1e-12:
        raise RuntimeError("Frequency grids differ between pol=0 and pol=90 job files.")

    c = 299_792_458.0
    lam_nm = (c / D0["f"]) * 1e9
    t_xx, t_xy, t_yx, t_yy = [np.asarray(v).reshape(-1) for v in (Ex_x0, Ex_y0, Ey_x0, Ey_y0)]
    lam_nm = np.asarray(lam_nm).reshape(-1)

    # build rows using your Jones module mapping
    rows_lin, rows_cir = [], []
    for i in range(lam_nm.size):
        conv = exy._from_linear_terms(t_xx[i], t_xy[i], t_yx[i], t_yy[i])
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
    return df_lin, df_cir

def queue_grid_jobs(fdtd,
                    l1_list_nm: Iterable[float],
                    l2_list_nm: Iterable[float],
                    l3_list_nm: Iterable[float],
                    cfg: ZSweepConfig,
                    job_dir: Path,
                    progress: bool = True
                    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parallel via Job Manager:
      For each (l1,l2,l3), create TWO job files: pol=0° and pol=90°.
      addjob(...), runjobs(), then harvest both files to build master DataFrames.
      Writes only master CSVs if cfg.save_master=True. No per-run CSVs.
    """
    job_dir = Path(job_dir); job_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    _set_solver(fdtd, cfg.sim_time_ps, cfg.auto_shutoff_level)

    l1_vals = list(l1_list_nm); l2_vals = list(l2_list_nm); l3_vals = list(l3_list_nm)
    total = len(l1_vals) * len(l2_vals) * len(l3_vals)

    # 1) Create & queue jobs
    pairs: List[Tuple[Path, Path, float, float, float]] = []
    k = 0
    for l1 in l1_vals:
        for l2 in l2_vals:
            for l3 in l3_vals:
                k += 1
                if progress:
                    print(f"[Z-sweep JM] prepare ({k}/{total}) l1={l1}, l2={l2}, l3={l3}")
                if fdtd.layoutmode() != 1:
                    fdtd.switchtolayout()
                _update_Z_script(fdtd, cfg.sg_name, l1, l2, l3, cfg.t_nm, cfg.t_z_nm, cfg.mat_z)

                # filenames per polarization
                base = f"Z_l1_{int(round(l1))}_l2_{int(round(l2))}_l3_{int(round(l3))}"
                f0 = job_dir / f"{base}_pol0.fsp"
                f90 = job_dir / f"{base}_pol90.fsp"

                # save two job files
                _save_jobfile(fdtd, f0, pol_deg=0.0,  cfg=cfg)
                _save_jobfile(fdtd, f90, pol_deg=90.0, cfg=cfg)

                # queue both jobs
                fdtd.addjob(str(f0))
                fdtd.addjob(str(f90))

                pairs.append((f0, f90, l1, l2, l3))

    # 2) run all queued jobs (in parallel if you have worker licenses)
    if progress:
        print(f"[Z-sweep JM] running {2*len(pairs)} jobs...")
    fdtd.runjobs()

    # 3) harvest results from each pair
    L_all: List[pd.DataFrame] = []
    C_all: List[pd.DataFrame] = []
    for idx, (f0, f90, l1, l2, l3) in enumerate(pairs, 1):
        if progress:
            print(f"[Z-sweep JM] harvest ({idx}/{len(pairs)}) {f0.name} + {f90.name}")
        df_lin, df_cir = _postprocess_two_files(fdtd, f0, f90, cfg)
        for df in (df_lin, df_cir):
            df["l1_nm"] = float(l1)
            df["l2_nm"] = float(l2)
            df["l3_nm"] = float(l3)
        L_all.append(df_lin); C_all.append(df_cir)

    dfL = pd.concat(L_all, ignore_index=True)
    dfC = pd.concat(C_all, ignore_index=True)

    if cfg.save_master:
        lin_path = out_dir / f"{cfg.master_prefix}_linear_ALL.csv"
        cir_path = out_dir / f"{cfg.master_prefix}_circular_ALL.csv"
        lin_path.write_text(dfL.to_csv(index=False))
        cir_path.write_text(dfC.to_csv(index=False))
        print(f"[Z-sweep JM] wrote:\n  {lin_path}\n  {cir_path}")

    return dfL, dfC

# ---------------- one-row-per-geometry summary ----------------
def summarize_circular(df_cir: pd.DataFrame) -> pd.DataFrame:
    """
    Make a compact, one-row-per-geometry summary from the circular-basis master DF.
    Columns returned (examples):
      - max_|T_LL|, lam_at_max_LL_nm
      - max_|T_RR|, lam_at_max_RR_nm
      - max_cross_amp (max over LR,RL), max_cross_dB, lam_at_max_cross_nm
      - CD_max = max(|T_RR|-|T_LL|) and its wavelength
      - avg_|T_LL|, avg_|T_RR|
    """
    def to_amp(x): return np.abs(np.asarray(x, dtype=complex))
    def to_db(m): return 20*np.log10(np.maximum(m, 1e-12))

    groups = df_cir.groupby(["l1_nm","l2_nm","l3_nm"], sort=False)
    rows = []
    for (l1,l2,l3), G in groups:
        lam = G["lambda_nm"].to_numpy()
        T_LL = G["T_LL"].to_numpy().astype(complex)
        T_RR = G["T_RR"].to_numpy().astype(complex)
        T_LR = G["T_LR"].to_numpy().astype(complex)
        T_RL = G["T_RL"].to_numpy().astype(complex)

        aLL, aRR = to_amp(T_LL), to_amp(T_RR)
        aLR, aRL = to_amp(T_LR), to_amp(T_RL)

        # co-pol peaks
        iLL = int(np.argmax(aLL)); iRR = int(np.argmax(aRR))
        maxLL, lamLL = float(aLL[iLL]), float(lam[iLL])
        maxRR, lamRR = float(aRR[iRR]), float(lam[iRR])

        # max cross-pol (take worst case of LR/RL)
        aX  = np.maximum(aLR, aRL)
        iX  = int(np.argmax(aX))
        maxX, lamX = float(aX[iX]), float(lam[iX])
        maxXdB = float(to_db(maxX))

        # circular dichroism (RR-LL)
        cd = aRR - aLL
        iCD = int(np.argmax(cd))
        CDmax, lamCD = float(cd[iCD]), float(lam[iCD])

        rows.append(dict(
            l1_nm=float(l1), l2_nm=float(l2), l3_nm=float(l3),
            max_TLL=maxLL, lam_at_max_TLL_nm=lamLL,
            max_TRR=maxRR, lam_at_max_TRR_nm=lamRR,
            max_cross_amp=maxX, max_cross_dB=maxXdB, lam_at_max_cross_nm=lamX,
            CD_max=CDmax, lam_at_CD_max_nm=lamCD,
            avg_TLL=float(aLL.mean()), avg_TRR=float(aRR.mean()),
        ))
    return pd.DataFrame(rows)
