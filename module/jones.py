# module/jones.py
from __future__ import annotations
import numpy as np
from numpy.linalg import eig, det

__all__ = [
    "linear_to_circular",
    "circular_to_linear",
    "ep_metrics",
    "from_linear_terms",
    "to_linear_terms",
]

def _U():
    # Columns are circular basis vectors in the linear basis:
    # |L>=(|x>+i|y>)/√2, |R>=(|x>-i|y>)/√2
    return (1/np.sqrt(2.0)) * np.array([[1, 1],
                                        [1j, -1j]], dtype=complex)

def linear_to_circular(T_lin: np.ndarray) -> np.ndarray:
    T = np.asarray(T_lin, dtype=complex)
    if T.shape[-2:] != (2, 2):
        raise ValueError("T_lin must be (..., 2, 2)")
    U = _U()
    Uinv = U.conj().T
    return Uinv @ T @ U

def circular_to_linear(T_cir: np.ndarray) -> np.ndarray:
    T = np.asarray(T_cir, dtype=complex)
    if T.shape[-2:] != (2, 2):
        raise ValueError("T_cir must be (..., 2, 2)")
    U = _U()
    Uinv = U.conj().T
    return U @ T @ Uinv

def ep_metrics(T: np.ndarray, tol_disc: float = 1e-3, tol_evec: float = 1e-2) -> dict:
    T = np.asarray(T, dtype=complex)
    tr = np.trace(T)
    delta = tr**2 - 4.0 * det(T)
    w, V = eig(T)
    split = abs(w[0] - w[1])
    s = np.linalg.svd(V, compute_uv=False)
    evec_cond = s[0] / max(s[1], 1e-12)
    nearly_degenerate = (abs(delta) <= tol_disc) or (split <= tol_disc)
    defective = evec_cond > (1.0 / tol_evec)
    is_ep = bool(nearly_degenerate and defective)
    is_diabolic = bool(nearly_degenerate and not defective)
    return {
        "eigvals": w,
        "discriminant": delta,
        "eigvec_cond": evec_cond,
        "is_ep": is_ep,
        "is_diabolic": is_diabolic,
        "split": split,
    }

def from_linear_terms(txx: complex, txy: complex, tyx: complex, tyy: complex) -> dict:
    """
   
      T_LL = 0.5*( txx + tyy + i*(txy - tyx) )
      T_LR = 0.5*( txx - tyy - i*(txy + tyx) )
      T_RL = 0.5*( txx - tyy + i*(txy + tyx) )
      T_RR = 0.5*( txx + tyy - i*(txy - tyx) )
    """
    T_lin = np.array([[txx, txy],
                      [tyx, tyy]], dtype=complex)
    i = 1j
    T_LL = 0.5*( txx + tyy + i*(txy - tyx) )
    T_LR = 0.5*( txx - tyy - i*(txy + tyx) )
    T_RL = 0.5*( txx - tyy + i*(txy + tyx) )
    T_RR = 0.5*( txx + tyy - i*(txy - tyx) )
    T_cir = np.array([[T_LL, T_LR],
                      [T_RL, T_RR]], dtype=complex)
    return {
        "T_lin": T_lin,
        "T_cir": T_cir,
        "T_LL": T_LL,
        "T_LR": T_LR,
        "T_RL": T_RL,
        "T_RR": T_RR,
    }

def to_linear_terms(T_LL: complex, T_LR: complex, T_RL: complex, T_RR: complex) -> dict:
    """
    Inverse mapping 
    """
    txx = 0.5 * ( (T_LL+T_RR) + (T_LR+T_RL) )
    tyy = 0.5 * ( (T_LL+T_RR) - (T_LR+T_RL) )
    txy = -(1j/2) * ( (T_LL - T_RR) + (T_RL - T_LR) )
    tyx = -(1j/2) * ( (T_RL - T_LR) - (T_LL - T_RR) )
    T_lin = np.array([[txx, txy],
                      [tyx, tyy]], dtype=complex)
    return {"T_lin": T_lin, "txx": txx, "txy": txy, "tyx": tyx, "tyy": tyy}
