# module/jones.py
# Utilities for Jones matrices, basis transforms, and EP diagnostics
from __future__ import annotations
import numpy as np
from numpy.linalg import eig, det

def linear_to_circular(T_lin: np.ndarray) -> np.ndarray:
    """
    Map 2x2 Jones matrix from linear {|x>,|y>} to circular {|L>,|R>} basis.
    Convention: |L> = (|x> + i|y>)/√2, |R> = (|x> - i|y>)/√2
    T_cir = U T_lin U^{-1}, where columns of U are basis vectors of new basis in old coordinates.
    """
    U = (1/np.sqrt(2.0)) * np.array([[1, 1], [1j, -1j]], dtype=complex)
    Uinv = np.conjugate(U.T)  # unitary
    return U @ T_lin @ Uinv

def circular_to_linear(T_cir: np.ndarray) -> np.ndarray:
    U = (1/np.sqrt(2.0)) * np.array([[1, 1], [1j, -1j]], dtype=complex)
    Uinv = np.conjugate(U.T)
    return Uinv @ T_cir @ U

def ep_metrics(T: np.ndarray, tol_disc: float = 1e-3, tol_evec: float = 1e-2) -> dict:
    """
    Diagnostics for an exceptional point in a 2x2 (complex) matrix.
    EP (2x2) heuristics:
      - Discriminant Δ = (tr T)^2 - 4 det T ≈ 0  (eigenvalue coalescence)
      - Defectiveness: only one independent eigenvector (Jordan block).
        We estimate by computing eigenvectors and checking linear dependence.
    Returns dict with eigenvalues, Δ, evec_condition, is_ep, is_diabolic.
    """
    tr = np.trace(T)
    delta = tr**2 - 4.0 * det(T)
    # eigen decomposition
    w, V = eig(T)
    # measure eigenvalue splitting
    split = min(abs(w[0]-w[1]), abs(w[1]-w[0]))
    # linear (in)dependence of eigenvectors: condition on [v1 v2]
    # If nearly collinear, rank ~ 1 -> defective
    # Use smallest singular value vs largest as a proxy
    s = np.linalg.svd(V, compute_uv=False)
    evec_cond = s[0] / max(s[1], 1e-12)
    nearly_degenerate = (abs(delta) <= tol_disc) or (split <= tol_disc)
    defective = evec_cond > (1.0 / tol_evec)  # huge condition -> nearly collinear
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
    Tlin = np.array([[txx, txy], [tyx, tyy]], dtype=complex)
    Tcir = linear_to_circular(Tlin)
    # Unpack circular basis (order: [L,R])
    T_LL, T_LR = Tcir[0, 0], Tcir[0, 1]
    T_RL, T_RR = Tcir[1, 0], Tcir[1, 1]
    return {
        "T_lin": Tlin,
        "T_cir": Tcir,
        "T_LL": T_LL,
        "T_LR": T_LR,
        "T_RL": T_RL,
        "T_RR": T_RR,
    }
