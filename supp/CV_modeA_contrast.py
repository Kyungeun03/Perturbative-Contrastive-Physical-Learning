"""
CV Photonics Classifier -- Mode A contrast training
==============================================================
Mode A: the target enters the update directly through the error e.

Per batch:
  1. free response        z_f^(n) = G(x_n, theta)
  2. K nudges             theta + beta * r_k,   r_k the rows of R (K x p)
  3. contrasts            dZ[n,k] = G(x_n, theta + beta r_k) - z_f^(n)
  4. update confined to span(R):  dtheta = R^T c

  Scalar form:
      c = dZ^T e
  Multivariate form:
      c = (dZ^T dZ + lam I)^-1 dZ^T (beta e)

Cost: K circuit evaluations per batch (plus N free responses), versus p evaluations per sample for the pseudoinverse Jacobian.

Set the parameters in the CONFIG block below.
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
import matplotlib.pyplot as plt

# ── Publication style ──────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'      : 'serif',
    'font.size'        : 11,
    'axes.labelsize'   : 12,
    'xtick.labelsize'  : 10,
    'ytick.labelsize'  : 10,
    'legend.fontsize'  : 10,
    'figure.dpi'       : 150,
    'axes.grid'        : False,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
})

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
BETA          = 0.05      # nudge magnitude
N_DIRECTIONS  = 8         # K: number of nudge directions per batch (< p = 12)
DIRECTION_RNG = 'rademacher'   # 'rademacher' (Spall) or 'gaussian'
REG           = 0.01      # lambda in the multivariate solve
RESAMPLE_R    = True      # draw new directions each batch

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
# Classifier circuit  (as in main/CV_pseudoinverse_multirun.py)
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
# Mode A: nudge, contrast, update
# ══════════════════════════════════════════════════════════════════════
def draw_directions(K, p):
    """Nudge directions, one per row."""
    if DIRECTION_RNG == 'rademacher':
        return np.random.choice([-1.0, 1.0], size=(K, p))
    elif DIRECTION_RNG == 'gaussian':
        return np.random.randn(K, p)
    raise ValueError(f"unknown DIRECTION_RNG: {DIRECTION_RNG}")


def measure_contrasts(X_batch, params, R):
    """
    Free responses and contrasts under each nudge direction.

    Returns
      z_f : (N,)   free response per sample
      dZ  : (N, K) contrast of sample n under direction k

    Circuit evaluations: N free + N*K nudged. The nudged parameter sets
    are shared across the batch, so only K distinct parameter vectors
    are ever prepared.
    """
    K = R.shape[0]
    z_f = np.array([cv_linear_classifier(x, params) for x in X_batch])
    dZ  = np.zeros((len(X_batch), K))
    for k in range(K):
        p_nudged = params + BETA * R[k]
        for n, x in enumerate(X_batch):
            dZ[n, k] = cv_linear_classifier(x, p_nudged) - z_f[n]
    return z_f, dZ


def mode_a_update(params, thresholds, X_batch, y_batch, eta, multivariate=True):
    """
    One Mode A update. The target sets the sign and size of the error that
    the contrasts are matched to, and the step is confined to the span of
    the nudge directions.
    """
    p = len(params)
    R = draw_directions(N_DIRECTIONS, p)

    z_f, dZ = measure_contrasts(X_batch, params, R)
    e = np.array([TARGET_VALUES[int(yt)] - zf for yt, zf in zip(y_batch, z_f)])

    if multivariate:
        A = dZ.T @ dZ + REG * np.eye(N_DIRECTIONS)
        try:
            c = np.linalg.solve(A, dZ.T @ (BETA * e))
        except np.linalg.LinAlgError:
            c = np.zeros(N_DIRECTIONS)
    else:
        # scalar form: correlation only, no inverse
        c = dZ.T @ e

    params = params + eta * (R.T @ c)

    # Adaptive threshold update (same rule as main/CV_pseudoinverse_multirun.py)
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
# Training loop
# ══════════════════════════════════════════════════════════════════════
def train(epochs=20, eta=0.1, batch_size=5, multivariate=True):
    params          = 0.1 * np.random.randn(N_PARAMS)
    thresholds      = [-0.5, 0.5]
    train_acc_hist  = []
    test_acc_hist   = []

    for epoch in range(epochs):
        idx    = np.random.permutation(len(X_train))
        X_shuf = X_train[idx]
        y_shuf = y_train[idx]

        for i in range(0, len(X_train), batch_size):
            Xb = X_shuf[i:i + batch_size]
            yb = y_shuf[i:i + batch_size].tolist()
            if len(Xb) == batch_size:
                params, thresholds = mode_a_update(
                    params, thresholds, Xb, yb, eta, multivariate)

        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            train_acc = accuracy_score(y_train, predict(X_train, params, thresholds))
            test_acc  = accuracy_score(y_test,  predict(X_test,  params, thresholds))
            train_acc_hist.append(train_acc)
            test_acc_hist.append(test_acc)
            print(f"  Epoch {epoch+1:3d}: train={train_acc:.3f}  test={test_acc:.3f}"
                  f"  thresholds={[round(t,3) for t in thresholds]}")

    return train_acc_hist, test_acc_hist, params, thresholds


def run_trials(n_trials=25, epochs=20, eta=0.1, batch_size=5, multivariate=True):
    all_test_hist, final_test = [], []
    for trial in range(n_trials):
        seed = 42 + trial * 13
        np.random.seed(seed)
        label = "multivariate" if multivariate else "scalar"
        print(f"\n--- Trial {trial+1}/{n_trials}  ({label}, seed={seed}) ---")
        _, test_hist, _, _ = train(epochs, eta, batch_size, multivariate)
        all_test_hist.append(test_hist)
        final_test.append(test_hist[-1])
        print(f"  -> final test acc: {test_hist[-1]:.1%}")
    return all_test_hist, np.array(final_test)


# ══════════════════════════════════════════════════════════════════════
# Main: scalar vs multivariate Mode A
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    N_TRIALS, EPOCHS, ETA, BATCH = 25, 20, 0.1, 5

    print("CV Photonics Classifier -- Mode A contrast training")
    print(f"beta={BETA}  K={N_DIRECTIONS}  directions={DIRECTION_RNG}  reg={REG}")
    print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

    print("\n" + "="*60)
    print("SCALAR contrast update  (c = dZ^T e)")
    print("="*60)
    _, final_scalar = run_trials(N_TRIALS, EPOCHS, ETA, BATCH, multivariate=False)

    print("\n" + "="*60)
    print("MULTIVARIATE contrast update  (c = (dZ^T dZ + lam I)^-1 dZ^T beta e)")
    print("="*60)
    _, final_multi = run_trials(N_TRIALS, EPOCHS, ETA, BATCH, multivariate=True)

    print("\n" + "="*60)
    print(f"SUMMARY ({N_TRIALS} trials, K={N_DIRECTIONS} of p={N_PARAMS})")
    print("="*60)
    print(f"  scalar       : {final_scalar.mean():.1%} +/- {final_scalar.std():.1%}")
    print(f"  multivariate : {final_multi.mean():.1%} +/- {final_multi.std():.1%}")
