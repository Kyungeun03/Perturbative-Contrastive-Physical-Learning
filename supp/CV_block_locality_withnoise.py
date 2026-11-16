"""
CV Photonics -- locality of the pseudoinverse update
====================================================
The full update solves one system over all p = 12 parameters. Here the
parameters are partitioned into blocks of size B, and each block is solved
using only its own columns of the Jacobian:

    d_theta[block] = (J_b^T J_b + lam I)^-1 J_b^T e

  B = 12 : the full update (one global solve)
  B < 12 : local, each block updated from its own responses
  B = 1  : fully local, one parameter at a time

Circuit, thresholds, data split and trial structure are unchanged from
main/CV_pseudoinverse_multirun.py. Only the solve is partitioned.

Results are appended to RESULTS_CSV after every trial, so an interrupted
run keeps whatever finished.
"""

import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import strawberryfields as sf
from strawberryfields import ops
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
BLOCK_SIZES  = [12, 6, 4, 2, 1]
NOISE_SIGMAS = [0.0, 0.001, 0.005, 0.01]   # input noise, redrawn per evaluation
N_TRIALS     = 25
EPOCHS       = 20
ETA          = 0.1
BATCH_SIZE   = 5
EPSILON      = 0.05      # finite-difference step, as in the main-text run
REG          = 0.01      # lambda
N_PARAMS     = 12
RESULTS_CSV  = "block_locality_results.csv"

# Noise is applied to the inputs only, during training only. The circuit and
# the readout are clean, and the test evaluation is noiseless, so the metric
# measures what was learned rather than the noise at test time.

TARGET_VALUES    = {0: -1.0, 1: 0.0, 2: 1.0}
CHECKPOINT_EVERY = 5

# ── Data ───────────────────────────────────────────────────────────────
iris = load_iris()
X, y = iris.data, iris.target
X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, stratify=y, random_state=42)


# ══════════════════════════════════════════════════════════════════════
# Circuit -- as in main/CV_pseudoinverse_multirun.py
# ══════════════════════════════════════════════════════════════════════
def cv_linear_classifier(x, params):
    eng  = sf.Engine("gaussian")
    prog = sf.Program(4)
    rotation_angles = params[:4]
    bs_params       = params[4:12]
    with prog.context as q:
        for i in range(4):
            ops.Dgate(x[i], 0) | q[i]
        for i in range(4):
            ops.Rgate(rotation_angles[i]) | q[i]
        ops.BSgate(bs_params[0], bs_params[1]) | (q[0], q[1])
        ops.BSgate(bs_params[2], bs_params[3]) | (q[2], q[3])
        ops.BSgate(bs_params[4], bs_params[5]) | (q[0], q[2])
        ops.BSgate(bs_params[6], bs_params[7]) | (q[1], q[3])
        ops.BSgate(np.pi / 4, 0) | (q[0], q[1])
        ops.BSgate(np.pi / 4, 0) | (q[0], q[3])
    return eng.run(prog).state.means()[0]


def predict(X, params, thresholds):
    preds = []
    for x in X:
        try:
            m = cv_linear_classifier(x, params)
            preds.append(0 if m < thresholds[0]
                         else (1 if m < thresholds[1] else 2))
        except Exception:
            preds.append(np.random.randint(0, 3))
    return np.array(preds)


def noisy(x, sigma):
    """Fresh input noise for every circuit evaluation."""
    return x if sigma == 0.0 else x + sigma * np.random.randn(*x.shape)


def compute_jacobian(x, params, sigma):
    base = cv_linear_classifier(noisy(x, sigma), params)
    J    = np.zeros(len(params))
    for i in range(len(params)):
        p_pert     = params.copy()
        p_pert[i] += EPSILON
        J[i]       = (cv_linear_classifier(noisy(x, sigma), p_pert)
                      - base) / EPSILON
    return J, base


# ══════════════════════════════════════════════════════════════════════
# Block-partitioned update
# ══════════════════════════════════════════════════════════════════════
def blocks_of(p, B):
    return [list(range(i, min(i + B, p))) for i in range(0, p, B)]


def block_update(params, thresholds, X_batch, y_batch, eta, B, sigma):
    jacobians, deltas, actuals = [], [], []
    for x, y_true in zip(X_batch, y_batch):
        J, base = compute_jacobian(x, params, sigma)
        jacobians.append(J)
        deltas.append(TARGET_VALUES[y_true] - base)
        actuals.append(base)

    J_mat = np.stack(jacobians)                 # (N, p)
    dv    = np.array(deltas).reshape(-1, 1)     # (N, 1)

    delta_p = np.zeros(len(params))
    for idx in blocks_of(len(params), B):
        Jb = J_mat[:, idx]                      # (N, |block|)
        A  = Jb.T @ Jb + REG * np.eye(len(idx))
        try:
            delta_p[idx] = (np.linalg.solve(A, Jb.T @ dv)).flatten()
        except np.linalg.LinAlgError:
            pass
    params = params + eta * delta_p

    # Adaptive threshold update -- unchanged
    actuals        = np.array(actuals)
    actual_classes = np.array(y_batch)
    damping        = 0.1
    mbc = [actuals[actual_classes == c] for c in range(3)]
    if len(mbc[0]) > 0 and len(mbc[1]) > 0:
        thresholds[0] = (1 - damping) * thresholds[0] \
                        + damping * (np.mean(mbc[0]) + np.mean(mbc[1])) / 2
    if len(mbc[1]) > 0 and len(mbc[2]) > 0:
        thresholds[1] = (1 - damping) * thresholds[1] \
                        + damping * (np.mean(mbc[1]) + np.mean(mbc[2])) / 2

    return params, thresholds


# ══════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════
def train(B, sigma):
    params     = 0.1 * np.random.randn(N_PARAMS)
    thresholds = [-0.5, 0.5]
    best, final = 0.0, 0.0

    for epoch in range(EPOCHS):
        idx    = np.random.permutation(len(X_train))
        X_shuf = X_train[idx]
        y_shuf = y_train[idx]

        for i in range(0, len(X_train), BATCH_SIZE):
            Xb = X_shuf[i:i + BATCH_SIZE]
            yb = y_shuf[i:i + BATCH_SIZE].tolist()
            if len(Xb) == BATCH_SIZE:
                params, thresholds = block_update(
                    params, thresholds, Xb, yb, ETA, B, sigma)

        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            acc  = accuracy_score(y_test, predict(X_test, params, thresholds))
            best = max(best, acc)
            final = acc
    return best, final


def main():
    new_file = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", buffering=1) as fh:
        if new_file:
            fh.write("sigma,block_size,trial,seed,best,final\n")

        summary = {}
        for sigma in NOISE_SIGMAS:
            for B in BLOCK_SIZES:
                bests, finals = [], []
                for t in range(N_TRIALS):
                    seed = 42 + t * 13
                    np.random.seed(seed)
                    b, f = train(B, sigma)
                    bests.append(b)
                    finals.append(f)
                    fh.write(f"{sigma},{B},{t},{seed},{b:.4f},{f:.4f}\n")
                    print(f"  sigma={sigma:<6g} B={B:>2}  "
                          f"trial {t+1:>2}/{N_TRIALS}  "
                          f"best={b:.3f}  final={f:.3f}", flush=True)
                summary[(sigma, B)] = (np.mean(finals), np.std(finals),
                                       np.mean(bests),  np.std(bests))
                m, s, bm, bs = summary[(sigma, B)]
                print(f"--> sigma={sigma:<6g} B={B:>2}  final {m:.1%}+-{s:.1%}"
                      f"   best {bm:.1%}+-{bs:.1%}\n", flush=True)

        for sigma in NOISE_SIGMAS:
            print("\n" + "=" * 58)
            print(f"input noise sigma = {sigma:g}")
            print(f"{'block':>6}  {'final':>18}  {'best':>18}")
            print("-" * 58)
            for B in BLOCK_SIZES:
                m, s, bm, bs = summary[(sigma, B)]
                print(f"{B:>6}  {m:>12.1%}+-{s:<5.1%}  {bm:>12.1%}+-{bs:<5.1%}")


if __name__ == "__main__":
    print("CV Photonics -- block locality of the pseudoinverse update")
    print(f"blocks={BLOCK_SIZES}  sigmas={NOISE_SIGMAS}  trials={N_TRIALS}  "
          f"epochs={EPOCHS}  eta={ETA}  reg={REG}  eps={EPSILON}")
    main()
