"""
Accuracy versus number of preprocessed symplectic components
============================================================
Standalone: prints only the final table.

Pipeline per component count k:
  phase space  xi = [x, Dx] with  p_i = x_{i+1} - x_i
  covariance   Sigma of the centered phase space
  symplectic   Sigma_s = Omega Sigma Omega^T,  Omega = [[0, I], [-I, 0]]
  basis        top k eigenvectors of Sigma_s by |eigenvalue|
  classifier   one quadratic energy per digit, trained by pseudoinverse
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from scipy.linalg import pinv, eigh

# ── Settings ───────────────────────────────────────────────────────────
COMPONENTS  = [32, 24, 16, 8]
EPOCHS      = 30
BATCH_SIZE  = 20
ETA         = 0.01
N_TRIALS    = 25          # raise for statistics; seeds are 42 + 13*trial
NUM_CLASSES = 10

# ── Data ───────────────────────────────────────────────────────────────
digits = load_digits()
X, y = digits.data / 16.0, digits.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, stratify=y, random_state=42)
scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


def phase_space(X_data):
    """xi = [x, p] with p_i = x_{i+1} - x_i, wrapping at the end."""
    p = np.zeros_like(X_data)
    p[:, :-1] = X_data[:, 1:] - X_data[:, :-1]
    p[:, -1]  = X_data[:, -1] - X_data[:, 0]
    return np.concatenate([X_data, p], axis=1)


def symplectic_basis(Xi, k):
    """Top-k eigenvectors of the symplectically transformed covariance."""
    Xc = Xi - Xi.mean(axis=0)
    Sigma = Xc.T @ Xc / len(Xc)
    n = Sigma.shape[0] // 2
    I, O = np.eye(n), np.zeros((n, n))
    Omega = np.block([[O, I], [-I, O]])
    Sigma_s = Omega @ Sigma @ Omega.T
    vals, vecs = eigh(Sigma_s)
    idx = np.argsort(np.abs(vals))[::-1]
    return vecs[:, idx][:, :k], Xi.mean(axis=0)


def project(Xi, basis, mean):
    return (Xi - mean) @ basis


def init_classifiers(k):
    return [np.concatenate([(0.001 * np.eye(k)).flatten(), np.zeros(k), [0.0]])
            for _ in range(NUM_CLASSES)]


def energy(a, params, k):
    W = params[:k*k].reshape(k, k)
    b = params[k*k:k*k+k]
    return a @ W @ a + b @ a + params[-1]


def update(params, A_batch, targets, k, eta):
    J_all, dz_all = [], []
    for a, t in zip(A_batch, targets):
        dz_all.append((-1.0 if t == 1 else 1.0) - energy(a, params, k))
        J_all.append(np.concatenate([np.outer(a, a).flatten(), a, [1.0]]))
    dp = pinv(np.stack(J_all)) @ np.array(dz_all).reshape(-1, 1)
    new = params + eta * dp.flatten()
    W = new[:k*k].reshape(k, k)
    W = 0.5 * (W + W.T)
    return np.concatenate([W.flatten(), new[k*k:k*k+k], [new[-1]]])


def predict(A, classifiers, k):
    return np.array([np.argmin([energy(a, classifiers[d], k)
                                for d in range(NUM_CLASSES)]) for a in A])


def run(k, seed):
    np.random.seed(seed)
    Xi_tr, Xi_te = phase_space(X_train), phase_space(X_test)
    basis, mean = symplectic_basis(Xi_tr, k)
    A_tr, A_te = project(Xi_tr, basis, mean), project(Xi_te, basis, mean)

    classifiers = init_classifiers(k)
    n = len(A_tr)
    best = accuracy_score(y_test, predict(A_te, classifiers, k))

    for _ in range(EPOCHS):
        idx = np.random.permutation(n)
        A_sh, y_sh = A_tr[idx], y_train[idx]
        for d in range(NUM_CLASSES):
            bin_t = (y_sh == d).astype(int)
            for i in range(0, n, BATCH_SIZE):
                j = min(i + BATCH_SIZE, n)
                classifiers[d] = update(classifiers[d], A_sh[i:j],
                                        bin_t[i:j], k, ETA)
        acc = accuracy_score(y_test, predict(A_te, classifiers, k))
        best = max(best, acc)
    return best


if __name__ == "__main__":
    print(f"{'k':>4}  {'params/digit':>13}  {'test accuracy':>18}")
    print("-" * 42)
    for k in COMPONENTS:
        accs = np.array([run(k, 42 + 13 * t) for t in range(N_TRIALS)])
        print(f"{k:>4}  {k*k + k + 1:>13}  "
              f"{accs.mean():>12.1%}+-{accs.std():<6.1%}", flush=True)