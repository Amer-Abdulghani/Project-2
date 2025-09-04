# scripts/sweep_two_run.py
# Two-run (Ex/Ey) Lumerical FDTD sweep -> Jones (linear) -> circular basis -> CSV
from __future__ import annotations
import os, sys, argparse, numpy as np, pandas as pd

def import_lumapi():
    """
    Import lumapi from the path specified in the LUMAPI_PATH environment variable.
    On Windows it often looks like:
      C:\Program Files\Lumerical\v241\api\python\lumapi.py
    """
    p = os.environ.get("LUMAPI_PATH", r"C:\Program Files\Lumerical\v241\api\python\lumapi.py")
    import importlib.util
    spec = importlib.util.spec_from_file_location("lumapi", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load lumapi from {p}. Set LUMAPI_PATH env var.")
    lumapi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lumapi)
    return lumapi

def area_average_complex(field_xyz):
    """
    Accepts arrays from getdata(mon, 'Ex'/'Ey').
    For a 2D DFT monitor, shape is typically (Nx, Ny, Nf).
    We average over spatial axes, keep frequency axis.
    """
    arr = np.array(field_xyz)
    while arr.ndim > 3:
        arr = arr.squeeze()
    if arr.ndim == 3:      # (Nx, Ny, Nf)
        return arr.mean(axis=(0, 1))
    elif arr.ndim == 2:    # (Nxy, Nf)
        return arr.mean(axis=0)
    else:                  # already (Nf,)
        return arr

def run_once(FDTD, pol_angle_deg: float, src_name: str, mon_name: str):
    """Set polarization, run, and pull Ex/Ey + frequency from a DFT monitor."""
    FDTD.switchtolayout()
    FDTD.setnamed(src_name, "polarization angle", float(pol_angle_deg))
    FDTD.run()
    f = np.array(FDTD.getdata(mon_name, "f"))   # Hz
    Ex = area_average_complex(FDTD.getdata(mon_name, "Ex"))
    Ey = area_average_complex(FDTD.getdata(mon_name, "Ey"))
    return f, Ex, Ey

def main():
    ap = argparse.ArgumentParser(description="Two-run Ex/Ey sweep → Jones → circular → CSV")
    ap.add_argument("--fsp", required=True, help="Path to .fsp project")
    ap.add_argument("--src", default="src", help="Source object name (e.g., 'src')")
    ap.add_argument("--mon", default="T", help="DFT monitor name on TRANSMISSION side (e.g., 'T')")
    ap.add_argument("--out", default="data/jones_spectra.csv", help="Output CSV file")
    args = ap.parse_args()

    # Make 'module' importable regardless of current working dir
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from module.jones import from_linear_terms

    lumapi = import_lumapi()
    with lumapi.FDTD(hide=True) as fdtd:
        fdtd.load(args.fsp)

        # Run #1: Ex incidence (0°)
        f, Ex_x, Ey_x = run_once(fdtd, 0.0, args.src, args.mon)
        # Run #2: Ey incidence (90°)
        f2, Ex_y, Ey_y = run_once(fdtd, 90.0, args.src, args.mon)

        if f.shape != f2.shape or np.max(np.abs(f - f2)) > 1e-9:
            raise RuntimeError("Frequency grids differ between Ex/Ey runs. Ensure identical monitor settings.")

        # Linear-basis Jones (area-averaged complex fields)
        t_xx, t_yx = Ex_x, Ey_x
        t_xy, t_yy = Ex_y, Ey_y

        # Frequency → wavelength (nm)
        c = 299_792_458.0
        lam_nm = (c / f) * 1e9

        # Convert each wavelength sample to circular basis and collect rows
        rows = []
        for i in range(len(lam_nm)):
            res = from_linear_terms(t_xx[i], t_xy[i], t_yx[i], t_yy[i])
            rows.append({
                "lambda_nm": lam_nm[i],
                "txx": t_xx[i], "txy": t_xy[i], "tyx": t_yx[i], "tyy": t_yy[i],
                "T_LL": res["T_LL"], "T_LR": res["T_LR"], "T_RL": res["T_RL"], "T_RR": res["T_RR"],
            })

        df = pd.DataFrame(rows).sort_values("lambda_nm")

        # Save both complex (as strings) and magnitude/phase columns
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        for col in ["txx","txy","tyx","tyy","T_LL","T_LR","T_RL","T_RR"]:
            df[col+"_mag"] = np.abs(df[col].values)
            df[col+"_ph_deg"] = np.angle(df[col].values, deg=True)
            df[col] = df[col].astype(str)

        df.to_csv(args.out, index=False)
        print(f"Wrote {args.out}  (rows: {len(df)})")

if __name__ == "__main__":
    main()
