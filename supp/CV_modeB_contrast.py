"""
CV Photonics Classifier -- Mode B by selection among candidate nudges
=====================================================================
The circuit, thresholds and trial structure are unchanged from
main/CV_pseudoinverse_multirun.py; only the parameter update differs.

Each probe is a candidate configuration; the measurement functional scores
candidates by their distance to the targets, and the update steps toward the
center of the selected set. The multivariate (elite-center) form averages over
the selected candidates instead of taking the single best.

Per batch:
  1. free responses   z_f[n] = G(x_n, theta)
  2. K candidates     theta_k = theta + beta r_k,  responses Z[n,k]
  3. score            s_k = sum_n ( z*_n - Z[n,k] )^2        (lower better)
  4. select           SCALAR : single best candidate
                      MULTI  : best ELITE, center weighted by 1/s
  5. update           theta <- theta + eta * beta * (selected direction)
"""

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
BETA          = 0.20      # candidate spread
ELITE         = 4         # candidates kept in the multivariate form
DIRECTION_RNG = 'rademacher'   # 'rademacher' or 'gaussian'
GATE_ON_IMPROVEMENT = True     # move only if the selection beats free

TARGET_VALUES    = {0: -1.0, 1: 0.0, 2: 1.0}
CHECKPOINT_EVERY = 5
N_PARAMS         = 12

# ── Data ───────────────────────────────────────────────────────────────
iris = load_iris()
X, y = iris.data, iris.target
X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, stratify=y, random_state=42)


# ══════════════════════════════════════════════════════════════════════
# Classifier circuit -- as in main/CV_pseudoinverse_multirun.py
# ══════════════════════════════════════════════════════════════════════
def cv_linear_classifier(x, params):
    """
    CV photonic classifier: 4 modes, displacement encoding,
    rotation + beamsplitter network, x-quadrature readout on mode 0.
    """
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


# ══════════════════════════════════════════════════════════════════════
# Mode B: candidate nudges, selection, center
# ══════════════════════════════════════════════════════════════════════
def draw_directions(K, p):
    if DIRECTION_RNG == 'rademacher':
        return np.random.choice([-1.0, 1.0], size=(K, p))
    if DIRECTION_RNG == 'gaussian':
        return np.random.randn(K, p)
    raise ValueError(f"unknown DIRECTION_RNG: {DIRECTION_RNG}")


def mode_b_select_update(params, thresholds, X_batch, y_batch,
                         eta, K, multivariate):
    R   = draw_directions(K, len(params))
    tgt = np.array([TARGET_VALUES[int(t)] for t in y_batch])

    # free responses
    z_f = np.array([cv_linear_classifier(x, params) for x in X_batch])

    # candidate responses: K distinct parameter vectors, shared across the batch
    Z = np.zeros((len(X_batch), K))
    for k in range(K):
        p_k = params + BETA * R[k]
        for n, x in enumerate(X_batch):
            Z[n, k] = cv_linear_classifier(x, p_k)

    # scoring rule -- squared distance of candidate outputs to targets
    s = ((tgt[:, None] - Z) ** 2).sum(axis=0)

    if multivariate:
        m   = min(ELITE, K)
        sel = np.argsort(s)[:m]
        w   = 1.0 / (s[sel] + 1e-12)
        w   = w / w.sum()
        step = (w[:, None] * R[sel]).sum(axis=0)
    else:
        sel  = np.array([int(np.argmin(s))])
        step = R[sel[0]]

    move = True
    if GATE_ON_IMPROVEMENT:
        move = s[sel].min() < ((tgt - z_f) ** 2).sum()
    if move:
        params = params + eta * BETA * step

    # Adaptive threshold update -- same rule as the Mode A script
    damping = 0.1
    yb  = np.asarray(y_batch)
    mbc = [z_f[yb == cls] for cls in range(3)]
    if len(mbc[0]) > 0 and len(mbc[1]) > 0:
        thresholds[0] = (1 - damping) * thresholds[0] \
                        + damping * (np.mean(mbc[0]) + np.mean(mbc[1])) / 2
    if len(mbc[1]) > 0 and len(mbc[2]) > 0:
        thresholds[1] = (1 - damping) * thresholds[1] \
                        + damping * (np.mean(mbc[1]) + np.mean(mbc[2])) / 2

    return params, thresholds


# ══════════════════════════════════════════════════════════════════════
# Training loop -- structure unchanged
# ══════════════════════════════════════════════════════════════════════
def train(epochs=20, eta=1.0, batch_size=5, K=8, multivariate=True):
    params     = 0.1 * np.random.randn(N_PARAMS)
    thresholds = [-0.5, 0.5]
    train_hist, test_hist = [], []

    for epoch in range(epochs):
        idx    = np.random.permutation(len(X_train))
        X_shuf = X_train[idx]
        y_shuf = y_train[idx]

        for i in range(0, len(X_train), batch_size):
            Xb = X_shuf[i:i + batch_size]
            yb = y_shuf[i:i + batch_size].tolist()
            if len(Xb) == batch_size:
                params, thresholds = mode_b_select_update(
                    params, thresholds, Xb, yb, eta, K, multivariate)

        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            tr = accuracy_score(y_train, predict(X_train, params, thresholds))
            te = accuracy_score(y_test,  predict(X_test,  params, thresholds))
            train_hist.append(tr)
            test_hist.append(te)
            print(f"  Epoch {epoch+1:3d}: train={tr:.3f}  test={te:.3f}"
                  f"  thresholds={[round(t,3) for t in thresholds]}")

    return train_hist, test_hist, params, thresholds


def run_trials(n_trials=5, epochs=20, eta=1.0, batch_size=5,
               K=8, multivariate=True):
    finals = []
    for trial in range(n_trials):
        seed = 42 + trial * 13
        np.random.seed(seed)
        label = "elite center" if multivariate else "best single"
        print(f"\n--- Trial {trial+1}/{n_trials}  ({label}, K={K}, seed={seed}) ---")
        _, test_hist, _, _ = train(epochs, eta, batch_size, K, multivariate)
        finals.append(test_hist[-1])
        print(f"  -> final test acc: {test_hist[-1]:.1%}")
    return np.array(finals)


if __name__ == "__main__":
    N_TRIALS, EPOCHS, ETA, BATCH, K = 25, 20, 1.0, 5, 8

    print("CV Photonics -- Mode B by selection among candidate nudges")
    print(f"beta={BETA}  K={K}  elite={ELITE}  gate={GATE_ON_IMPROVEMENT}")
    print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

    print("\n" + "="*60)
    print("SCALAR  (single best candidate)")
    print("="*60)
    f_scalar = run_trials(N_TRIALS, EPOCHS, ETA, BATCH, K, multivariate=False)

    print("\n" + "="*60)
    print("MULTIVARIATE  (elite center)")
    print("="*60)
    f_multi = run_trials(N_TRIALS, EPOCHS, ETA, BATCH, K, multivariate=True)

    print("\n" + "="*60)
    print(f"SUMMARY ({N_TRIALS} trials)")
    print("="*60)
    print(f"  best single  : {f_scalar.mean():.1%} +/- {f_scalar.std():.1%}")
    print(f"  elite center : {f_multi.mean():.1%} +/- {f_multi.std():.1%}")