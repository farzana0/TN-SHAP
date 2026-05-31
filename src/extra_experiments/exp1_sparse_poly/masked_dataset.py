import numpy as np

def chebyshev_nodes_01(m: int) -> np.ndarray:
    """Chebyshev–Gauss nodes mapped to [0,1]."""
    if m <= 0:
        return np.zeros((0,), dtype=np.float32)
    k = np.arange(m, dtype=np.float64)
    nodes = np.cos((2 * k + 1) * np.pi / (2 * m))
    t = (nodes + 1.0) * 0.5
    return t.astype(np.float32)


def build_masked_dataset_k1(Xtargets: np.ndarray, t_nodes: np.ndarray):
    """
    Build masked dataset for k=1 only.

    For each target x (K of them), each feature i, each t in t_nodes:
      base = t * x
      x1   = base           # feature i kept
      x0   = base with x_i=0

    Returns:
        masked_X: [M, D] float32
    """
    Xtargets = np.asarray(Xtargets, np.float32)
    t_nodes = np.asarray(t_nodes, np.float32)

    K, D = Xtargets.shape
    masked_rows = []

    for r in range(K):
        x = Xtargets[r]
        for i in range(D):
            for t in t_nodes:
                base = (t * x).astype(np.float32)
                x1 = base.copy()
                x0 = base.copy()
                x0[i] = 0.0
                masked_rows.extend([x1, x0])

    masked_X = np.stack(masked_rows, axis=0).astype(np.float32)
    return masked_X
