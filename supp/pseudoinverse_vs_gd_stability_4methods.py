"""
Multivariate update rules vs GD: stability comparison
=====================================================
Four update rules compared under the same Mode A loop:

  GD                first-order, per-sample
  Gauss-Newton      pinv(J) @ dm                       (lambda = 0)
  Tikhonov          inv(J^T J + lam I) J^T @ dm        (lambda fixed)
  Levenberg-Marquardt  same form, lambda decayed per epoch

Two experiments:
1. Eta sweep: vary the learning rate
2. Ill-conditioned features: scale the features unevenly

Both use 1-output linear model, fixed thresholds, no adaptive threshold.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
np.seterr(over='ignore', invalid='ignore')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

plt.rcParams.update({
    'font.family'      : 'serif',
    'font.size'        : 11,
    'axes.labelsize'   : 12,
    'xtick.labelsize'  : 10,
    'ytick.labelsize'  : 10,
    'legend.fontsize'  : 9,
    'figure.dpi'       : 150,
    'axes.grid'        : False,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
})

# ── Data ───────────────────────────────────────────────────────────────
iris = load_iris()
X, y = iris.data, iris.target
X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, stratify=y, random_state=42)

TARGET_VALUES = {0: -1.0, 1: 0.0, 2: 1.0}
THRESHOLDS    = [-0.5, 0.5]
EPOCHS        = 20
N_TRIALS      = 25
BATCH_SIZE    = 5
REG           = 0.01      # fixed lambda for Tikhonov
LM_LAM0       = 1.0       # initial lambda for LM
LM_DECAY      = 0.6       # lambda <- LM_DECAY * lambda each epoch

def predict(X, w, b):
    return np.array([0 if w@x+b < THRESHOLDS[0]
                     else (1 if w@x+b < THRESHOLDS[1] else 2)
                     for x in X])

# ── Training functions ─────────────────────────────────────────────────
def train_gd(seed, eta, X_tr=None, X_te=None):
    if X_tr is None: X_tr, X_te = X_train, X_test
    np.random.seed(seed)
    w = 0.1 * np.random.randn(4)
    b = 0.0
    best_acc, best_w, best_b = 0, w.copy(), b
    hist = []
    for epoch in range(EPOCHS):
        idx = np.random.permutation(len(X_tr))
        for i in idx:
            x, y_true = X_tr[i], int(y_train[i])
            error = TARGET_VALUES[y_true] - (w @ x + b)
            w += eta * error * x
            b += eta * error
        acc = accuracy_score(y_train, predict(X_tr, w, b))
        if acc > best_acc:
            best_acc, best_w, best_b = acc, w.copy(), b
        hist.append(accuracy_score(y_test, predict(X_te, best_w, best_b)))
    return hist


def _train_multivariate(seed, eta, X_tr, X_te, solver):
    """Shared Mode A loop; `solver(J, dm, epoch)` returns the update vector."""
    if X_tr is None: X_tr, X_te = X_train, X_test
    np.random.seed(seed)
    w = 0.1 * np.random.randn(4)
    b = 0.0
    best_acc, best_w, best_b = 0, w.copy(), b
    hist = []
    for epoch in range(EPOCHS):
        idx = np.random.permutation(len(X_tr))
        for i in range(0, len(X_tr), BATCH_SIZE):
            batch = idx[i:i+BATCH_SIZE]
            if len(batch) != BATCH_SIZE: continue
            Xb = X_tr[batch]
            yb = y_train[batch]
            J  = np.column_stack([Xb, np.ones(BATCH_SIZE)])
            dm = np.array([TARGET_VALUES[int(yi)] - (w@x+b)
                           for x, yi in zip(Xb, yb)])
            try:
                delta = solver(J, dm, epoch)
                w += eta * delta[:4]
                b += eta * delta[4]
            except np.linalg.LinAlgError:
                pass
        acc = accuracy_score(y_train, predict(X_tr, w, b))
        if acc > best_acc:
            best_acc, best_w, best_b = acc, w.copy(), b
        hist.append(accuracy_score(y_test, predict(X_te, best_w, best_b)))
    return hist


def train_gauss_newton(seed, eta, X_tr=None, X_te=None):
    """lambda = 0: undamped pseudoinverse."""
    return _train_multivariate(
        seed, eta, X_tr, X_te,
        lambda J, dm, epoch: np.linalg.pinv(J) @ dm)


def train_tikhonov(seed, eta, X_tr=None, X_te=None):
    """Fixed damping."""
    def solver(J, dm, epoch):
        Jt = J.T
        return np.linalg.inv(Jt@J + REG*np.eye(5)) @ Jt @ dm
    return _train_multivariate(seed, eta, X_tr, X_te, solver)


def train_lm(seed, eta, X_tr=None, X_te=None):
    """Levenberg-Marquardt with a decaying damping schedule:
    gradient-like early (large lambda), Gauss-Newton-like late (small lambda)."""
    def solver(J, dm, epoch):
        lam = LM_LAM0 * (LM_DECAY ** epoch)
        Jt = J.T
        return np.linalg.inv(Jt@J + lam*np.eye(5)) @ Jt @ dm
    return _train_multivariate(seed, eta, X_tr, X_te, solver)


METHODS = [
    ('GD',        train_gd,            'steelblue', 'o', '-'),
    ('PINV',      train_gauss_newton,  'darkorange','^', '-'),
    ('REGPINV',   train_tikhonov,      'firebrick', 's', '-'),
    ('LM',        train_lm,            'seagreen',  'D', '-'),
]
# legend labels, matching Supplementary Note 2 terminology
LABELS = {
    'GD'     : 'GD',
    'PINV'   : r'Pseudoinverse ($\lambda=0$)',
    'REGPINV': r'Regularized pseudoinverse ($\lambda$ fixed)',
    'LM'     : r'Levenberg-Marquardt ($\lambda$ decaying)',
}
# short labels for the printed tables
SHORT = {'GD': 'GD', 'PINV': 'pinv (l=0)',
         'REGPINV': 'reg pinv (l fix)', 'LM': 'LM (l decay)'}

# ══════════════════════════════════════════════════════════════════════
# Experiment 1: Eta sweep
# ══════════════════════════════════════════════════════════════════════
eta_vals = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
seeds    = [42 + i*13 for i in range(N_TRIALS)]

results_eta = {name: [] for name, *_ in METHODS}

print("Experiment 1: Learning-rate sweep")
header = f"{'eta':>8}  " + "  ".join(f"{SHORT[n]:>18}" for n, *_ in METHODS)
print(header)
print("-" * len(header))
for eta in eta_vals:
    row = []
    for name, fn, *_ in METHODS:
        accs = [fn(s, eta)[-1] for s in seeds]
        results_eta[name].append((np.mean(accs), np.std(accs)))
        row.append(f"{np.mean(accs):>12.1%}+-{np.std(accs):<7.1%}")
    print(f"{eta:>8.3f}  " + "  ".join(row))

# ══════════════════════════════════════════════════════════════════════
# Experiment 2: Ill-conditioned features
# ══════════════════════════════════════════════════════════════════════
scale_factors = [
    (1.0,  1.0,  1.0,  1.0),   # well-conditioned
    (10.0, 1.0,  1.0,  0.1),   # mild ill-conditioning
    (100., 1.0,  1.0,  0.01),  # strong ill-conditioning
    (1e3,  1.0,  1.0,  1e-3),  # severe ill-conditioning
]

print("\nExperiment 2: Ill-conditioned features")
header = f"{'scale':>18}  " + "  ".join(f"{SHORT[n]:>18}" for n, *_ in METHODS)
print(header)
print("-" * len(header))

results_ill = {name: [] for name, *_ in METHODS}
labels_ill = []
eta = 0.1
for scales in scale_factors:
    S = np.array(scales)
    X_tr_ill = X_train * S
    X_te_ill = X_test  * S
    label = f"({scales[0]:.0e},{scales[3]:.0e})"
    labels_ill.append(label)
    row = []
    for name, fn, *_ in METHODS:
        accs = [fn(s, eta, X_tr_ill, X_te_ill)[-1] for s in seeds]
        results_ill[name].append((np.mean(accs), np.std(accs)))
        row.append(f"{np.mean(accs):>12.1%}+-{np.std(accs):<7.1%}")
    print(f"{label:>18}  " + "  ".join(row))

# ══════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5))

# Panel (a): eta sweep
ax = axes[0]
x = np.arange(len(eta_vals))
for name, _fn, color, marker, ls in METHODS:
    mean = [v[0] for v in results_eta[name]]
    std  = [v[1] for v in results_eta[name]]
    ax.errorbar(x, mean, yerr=std, color=color, linewidth=1.6, linestyle=ls,
                marker=marker, markersize=4.5, capsize=3,
                label=LABELS.get(name, name))
ax.set_xticks(x)
ax.set_xticklabels([str(e) for e in eta_vals], fontsize=9)
ax.set_xlabel(r'Learning rate $\eta$')
ax.set_ylabel('Test accuracy')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_: f'{y:.0%}'))
ax.legend(frameon=False, loc='lower left')
ax.text(0.04, 0.97, '(a)', transform=ax.transAxes,
        ha='left', va='top', fontsize=11, fontweight='bold')

# Panel (b): ill-conditioned features
ax = axes[1]
x = np.arange(len(labels_ill))
for name, _fn, color, marker, ls in METHODS:
    mean = [v[0] for v in results_ill[name]]
    std  = [v[1] for v in results_ill[name]]
    ax.errorbar(x, mean, yerr=std, color=color, linewidth=1.6, linestyle=ls,
                marker=marker, markersize=4.5, capsize=3,
                label=LABELS.get(name, name))
ax.set_xticks(x)
ax.set_xticklabels(labels_ill, fontsize=9)
ax.set_xlabel(r'Feature scaling $(s_0, s_3)$')
ax.set_ylabel('Test accuracy')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_: f'{y:.0%}'))
ax.legend(frameon=False, loc='lower left')
ax.text(0.04, 0.97, '(b)', transform=ax.transAxes,
        ha='left', va='top', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('pseudoinverse_vs_gd_stability.svg', bbox_inches='tight')
plt.savefig('pseudoinverse_vs_gd_stability.png', bbox_inches='tight', dpi=200)
plt.close()
print("\nSaved pseudoinverse_vs_gd_stability.svg/.png")