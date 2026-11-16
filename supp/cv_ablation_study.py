"""
Ablation study: CV photonics classifier
========================================
Four configurations isolating the contribution of each component:

  Config A -- CV multiplier grad  + CV displacement update  (full CV system)
  Config B -- Classical grad      + CV displacement update  (isolates gradient error)
  Config C -- CV multiplier grad  + classical GD update     (isolates update error)
  Config D -- Classical grad      + classical GD update     (full classical baseline)

All configurations use:
  - Identical classifier circuit (cv_linear_classifier)
  - Input scaling (-1, 1)
  - epsilon = 0.05
  - batch_size = 1 (online learning)
  - 25 independent trials, 20 epochs
  - Evaluation on held-out test set
"""

import numpy as np
import strawberryfields as sf
from strawberryfields import ops
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message=".*disconnected components.*")

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

COLORS = {
    'A': '#2563EB',   # blue   — full CV
    'B': '#EA580C',   # orange — classical grad + CV update
    'C': '#16A34A',   # green  — CV grad + classical GD
    'D': '#DC2626',   # red    — classical grad + classical GD
}
LABELS = {
    'A': 'Config A: CV grad + CV update (batch=1)',
    'B': 'Config B: Classical grad + CV update (batch=1)',
    'C': 'Config C: CV grad + classical GD (batch=1)',
    'D': 'Config D: Classical grad + classical GD (batch=1)',
}

# ══════════════════════════════════════════════════════════════════════
# Shared classifier circuit  (identical across all 4 configs)
# ══════════════════════════════════════════════════════════════════════
def cv_linear_classifier(x, params):
    """
    CV photonic classifier: 4 modes, displacement encoding,
    rotation + beamsplitter network, homodyne readout on mode 0.
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

    result = eng.run(prog)
    return result.state.means()[0]   # X quadrature, mode 0


def predict(X, params, thresholds):
    preds = []
    for x in X:
        try:
            m = cv_linear_classifier(x, params)
            if m < thresholds[0]:
                preds.append(0)
            elif m < thresholds[1]:
                preds.append(1)
            else:
                preds.append(2)
        except Exception:
            preds.append(np.random.randint(0, 3))
    return np.array(preds)


# ══════════════════════════════════════════════════════════════════════
# Gradient methods
# ══════════════════════════════════════════════════════════════════════
EPSILON = 0.05

def classical_gradient(x, params, target_values, y_true):
    """
    Classical finite-difference Jacobian for one sample.
    Returns (J, base_measurement) where J[i] = df/dtheta_i.
    Pure Jacobian -- error signal handled by the update function.
    """
    base = cv_linear_classifier(x, params)
    J    = np.zeros(len(params))
    for i in range(len(params)):
        p_pert    = params.copy()
        p_pert[i] += EPSILON
        J[i]      = (cv_linear_classifier(x, p_pert) - base) / EPSILON
    return J, base


def cv_multiplier_gradient(x, params, target_values, y_true):
    """
    CV multiplier gradient: finite difference via intensity-difference
    homodyne circuit.  Returns (J, base_measurement).
    Physically: delta_f * e recovered at detector level via
        (alpha1^2 - alpha0^2) / 2 = delta_f * e
    where alpha_k = x_k_mean / 2  (SF hbar=2 convention).
    """
    base = cv_linear_classifier(x, params)
    target = target_values[y_true]
    error  = target - base

    J = np.zeros(len(params))
    for i in range(len(params)):
        p_pert = params.copy()
        p_pert[i] += EPSILON
        pert   = cv_linear_classifier(x, p_pert)
        delta_f = pert - base

        if abs(delta_f) < 1e-12 and abs(error) < 1e-12:
            J[i] = 0.0
            continue

        # CV multiplier circuit
        eng  = sf.Engine("gaussian")
        prog = sf.Program(2)
        with prog.context as q:
            ops.Dgate(delta_f, 0) | q[0]
            ops.Dgate(error,   0) | q[1]
            ops.BSgate(np.pi / 4, 0) | (q[0], q[1])
        result = eng.run(prog)
        means  = result.state.means()
        alpha0 = means[0] / 2.0
        alpha1 = means[1] / 2.0
        # CV multiplier gives delta_f * error / epsilon
        # which equals df/dtheta_i * error -- the error-weighted Jacobian
        product = (alpha1**2 - alpha0**2) / 2.0
        J[i]    = product / EPSILON   # = (df/dtheta_i) * error / epsilon

    return J, base


# ══════════════════════════════════════════════════════════════════════
# Update methods
# ══════════════════════════════════════════════════════════════════════
def pseudoinverse_update(params, thresholds, X_batch, y_batch,
                         target_values, eta, grad_fn):
    """
    Regularised pseudoinverse parameter update.

    Handles two cases depending on grad_fn:
    - classical_gradient: returns pure Jacobian J (df/dtheta).
      Update: delta_p = (J^T J + reg I)^-1 J^T delta_m
      where delta_m = target - base (error vector).
    - cv_multiplier_gradient: returns error-weighted Jacobian J*e/eps.
      Error is already encoded, so use J directly as update direction:
      delta_p = (J^T J + reg I)^-1 J^T ones
    """
    jacobians           = []
    delta_measurements  = []
    actual_measurements = []
    is_cv_grad = (grad_fn.__name__ == 'cv_multiplier_gradient')

    for x, y_true in zip(X_batch, y_batch):
        J, base = grad_fn(x, params, target_values, y_true)
        if J is not None:
            jacobians.append(J)
            if is_cv_grad:
                delta_measurements.append(1.0)   # error already in J
            else:
                delta_measurements.append(target_values[y_true] - base)
            actual_measurements.append(base)

    if not jacobians:
        return params, thresholds

    J_mat = np.stack(jacobians)
    dv    = np.array(delta_measurements).reshape(-1, 1)
    reg   = 0.01
    Jt    = J_mat.T
    try:
        delta_p = np.linalg.inv(Jt @ J_mat + reg * np.eye(Jt.shape[0])) @ Jt @ dv
        params  = params + eta * delta_p.flatten()
    except np.linalg.LinAlgError:
        pass

    # Adaptive threshold update
    actual_measurements = np.array(actual_measurements)
    actual_classes      = np.array(y_batch)
    damping = 0.1
    mbc = [actual_measurements[actual_classes == c] for c in range(3)]
    if len(mbc[0]) > 0 and len(mbc[1]) > 0:
        new_tau0 = (np.mean(mbc[0]) + np.mean(mbc[1])) / 2
        thresholds[0] = (1 - damping) * thresholds[0] + damping * new_tau0
    if len(mbc[1]) > 0 and len(mbc[2]) > 0:
        new_tau1 = (np.mean(mbc[1]) + np.mean(mbc[2])) / 2
        thresholds[1] = (1 - damping) * thresholds[1] + damping * new_tau1

    return params, thresholds


def cv_circuit_update(params, thresholds, X_batch, y_batch,
                      target_values, eta, grad_fn):
    """
    CV displacement update circuit.

    Each parameter mode is initialised with the current parameter value
    via a displacement gate, then the scaled gradient is added via a
    second displacement gate.  The updated parameter is read from the
    mode mean (SF hbar=2: divide by 2):

        D(theta_k) D(eta * J_k) |0>  =>  <x> / 2 = theta_k + eta * J_k

    Gradients are averaged over the batch before the circuit is run.
    """
    n_params = len(params)

    # Accumulate error-weighted gradients over batch
    # For CV multiplier grad: J already contains delta_f * error / eps
    # For classical grad: J is pure Jacobian, so multiply by error here
    batch_grads = []
    for x, y_true in zip(X_batch, y_batch):
        J, base = grad_fn(x, params, target_values, y_true)
        if J is not None:
            error = target_values[y_true] - base
            # cv_multiplier_gradient already encodes error; classical_gradient does not
            # Detected by checking grad_fn name
            if grad_fn.__name__ == 'classical_gradient':
                J = J * error
            batch_grads.append(J)

    if not batch_grads:
        return params, thresholds

    mean_grad = np.mean(batch_grads, axis=0)

    try:
        eng  = sf.Engine("gaussian")
        prog = sf.Program(n_params)

        with prog.context as q:
            for k in range(n_params):
                ops.Dgate(float(params[k]),          0) | q[k]  # current parameter
                ops.Dgate(float(eta * mean_grad[k]), 0) | q[k]  # scaled gradient

        result = eng.run(prog)
        means  = result.state.means()   # [x0,...,x_{n-1}, p0,...,p_{n-1}]

        # x-quadrature means in positions [0..n_params-1], hbar=2: divide by 2
        new_params = np.array([means[k] / 2.0 for k in range(n_params)])
        params = np.clip(new_params, -1.8, 1.8)

    except Exception as e:
        print(f"  CV update circuit failed: {e}")

    return params, thresholds


def classical_gd_update(params, thresholds, X_batch, y_batch,
                        target_values, eta, grad_fn):
    """
    Classical gradient descent update.
    Computes error-weighted gradient per sample, averages over batch,
    updates params arithmetically: params += eta * mean(grad).
    Compatible with batch_size=1 (online learning).
    """
    batch_grads = []
    for x, y_true in zip(X_batch, y_batch):
        J, base = grad_fn(x, params, target_values, y_true)
        if J is not None:
            # J from cv_multiplier_gradient already encodes error
            # J from classical_gradient is pure Jacobian -- multiply by error
            if grad_fn.__name__ == 'classical_gradient':
                error = target_values[y_true] - base
                J = J * error
            batch_grads.append(J)

    if batch_grads:
        params = params + eta * np.mean(batch_grads, axis=0)
        params = np.clip(params, -1.8, 1.8)

    return params, thresholds


# ══════════════════════════════════════════════════════════════════════
# Generic training loop
# ══════════════════════════════════════════════════════════════════════
TARGET_VALUES = {0: -1.0, 1: 0.0, 2: 1.0}
CHECKPOINT_EVERY = 1


def cv_train(X_train, y_train, X_test, y_test,
             grad_fn, update_fn,
             epochs=20, eta=0.1, batch_size=1):
    """
    Train CVPhotonicsClassifier with specified gradient and update functions.
    Evaluates on X_test at each checkpoint.
    Returns (train_acc_history, test_acc_history,
             best_params, best_thresholds, final_params, final_thresholds).
    """
    params          = 0.1 * np.random.randn(12)
    thresholds      = [-0.5, 0.5]
    train_acc_hist  = []
    test_acc_hist   = []
    best_acc        = 0.0
    best_params     = params.copy()
    best_thresholds = list(thresholds)

    for epoch in range(epochs):
        idx       = np.random.permutation(len(X_train))
        X_shuf    = X_train[idx]
        y_shuf    = y_train[idx]

        for i in range(0, len(X_train), batch_size):
            Xb = X_shuf[i:i + batch_size]
            yb = y_shuf[i:i + batch_size]
            if len(Xb) == batch_size:
                params, thresholds = update_fn(
                    params, thresholds, Xb, yb.tolist(),
                    TARGET_VALUES, eta, grad_fn)

        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            train_preds = predict(X_train, params, thresholds)
            test_preds  = predict(X_test,  params, thresholds)
            train_acc   = accuracy_score(y_train, train_preds)
            test_acc    = accuracy_score(y_test,  test_preds)
            train_acc_hist.append(train_acc)
            test_acc_hist.append(test_acc)
            print(f"    Epoch {epoch+1:3d}: train={train_acc:.3f}  test={test_acc:.3f}")
            if test_acc > best_acc:
                best_acc        = test_acc
                best_params     = params.copy()
                best_thresholds = list(thresholds)

    return (train_acc_hist, test_acc_hist,
            best_params, best_thresholds, params, thresholds)


# ══════════════════════════════════════════════════════════════════════
# Ablation runner
# ══════════════════════════════════════════════════════════════════════
CONFIGS = {
    'A': (cv_multiplier_gradient,  cv_circuit_update),
    'B': (classical_gradient,      cv_circuit_update),
    'C': (cv_multiplier_gradient,  classical_gd_update),
    'D': (classical_gradient,      classical_gd_update),
}

def run_ablation(X_train, X_test, y_train, y_test,
                 n_trials=5, epochs=20, eta=0.1, batch_size=1):
    results = {k: {'hist': [], 'train_hist': [],
                    'final_params': [], 'final_thresholds': []}
               for k in CONFIGS}

    for cfg_key, (grad_fn, update_fn) in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  {LABELS[cfg_key]}")
        print(f"{'='*60}")
        for trial in range(n_trials):
            seed = 42 + trial * 13
            np.random.seed(seed)
            print(f"  Trial {trial+1}/{n_trials}  (seed={seed})")
            train_hist, test_hist, best_p, best_t, final_p, final_t = cv_train(
                X_train, y_train, X_test, y_test,
                grad_fn, update_fn,
                epochs=epochs, eta=eta, batch_size=batch_size)
            results[cfg_key]['hist'].append(test_hist)
            results[cfg_key]['train_hist'].append(train_hist)
            results[cfg_key]['final_params'].append(final_p)
            results[cfg_key]['final_thresholds'].append(final_t)

    return results


# ══════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════
def plot_ablation(results, epochs, checkpoint_every=5):
    """
    Ablation plot.
    4 panels side by side, shared y-axis, no title, no grid.
    Each panel annotates: gradient method, update method, batch scheme.

    Batch scheme notes:
      A/B (CV circuit update): gradients averaged over batch, one circuit call per batch
      C/D (pseudoinverse update): full minibatch pseudoinverse solve per batch
    """
    checkpoint_epochs = np.arange(checkpoint_every, epochs + 1, checkpoint_every)

    # Component and batch scheme labels per config
    INFO = {
        'A': ('CV multiplier grad', 'CV circuit update',  'batch=1'),
        'B': ('Classical grad',     'CV circuit update',  'batch=1'),
        'C': ('CV multiplier grad', 'Classical GD',       'batch=1'),
        'D': ('Classical grad',     'Classical GD',       'batch=1'),
    }

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True)

    for ax, cfg_key in zip(axes, ['A', 'B', 'C', 'D']):
        color     = COLORS[cfg_key]
        histories = results[cfg_key]['hist']
        max_len   = max(len(h) for h in histories)
        padded    = np.array([
            np.pad(h, (0, max_len - len(h)), mode='edge')
            for h in histories
        ])
        mean_acc = padded.mean(axis=0)
        std_acc  = padded.std(axis=0)
        ep_x     = checkpoint_epochs[:max_len]

        # Individual trial curves
        for curve in padded:
            ax.plot(ep_x, curve, color=color, alpha=0.15, linewidth=0.8)

        # Mean +/- std band
        ax.fill_between(ep_x,
                        mean_acc - std_acc,
                        mean_acc + std_acc,
                        color=color, alpha=0.20)

        # Mean curve (final)
        ax.plot(ep_x, mean_acc, color=color, linewidth=2.0,
                marker='o', markersize=4,
                label=f'Mean(final) = {mean_acc[-1]:.1%}')

        # Best accuracy per trial as dashed horizontal line
        best_accs = np.array([max(h) for h in histories])
        ax.axhline(best_accs.mean(), color=color, linestyle='--',
                   linewidth=1.4, alpha=0.8,
                   label=f'Mean(best) = {best_accs.mean():.1%}')

        # Random baseline
        ax.axhline(1/3, color='gray', linestyle=':', linewidth=1.0,
                   alpha=0.6, label='Random (33%)')

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylim(0.10, 1.05)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.tick_params(direction='in', top=False, right=False)
        ax.legend(frameon=False, fontsize=9, loc='lower right')

        # Config label top-left
        ax.text(0.04, 0.97, f'({cfg_key})',
                transform=ax.transAxes, ha='left', va='top',
                fontsize=11, fontweight='bold')



    axes[0].set_ylabel('Training Accuracy', fontsize=12)
    fig.tight_layout()
    fig.savefig('cv_ablation_training_curves.svg', bbox_inches='tight')
    fig.savefig('cv_ablation_training_curves.png', bbox_inches='tight', dpi=200)
    print("Saved cv_ablation_training_curves.svg/.png")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Data — consistent with pseudoinverse baseline
    iris = load_iris()
    X, y = iris.data, iris.target
    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=42)

    print("CV Photonics Classifier — Ablation Study")
    print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")
    print(f"Configs: A (full CV), B (classical grad + CV update), "
          f"C (CV grad + pseudoinverse), D (full pseudoinverse)")

    N_TRIALS   = 25
    EPOCHS     = 20
    ETA        = 0.1
    BATCH_SIZE = 1

    results = run_ablation(
        X_train, X_test, y_train, y_test,
        n_trials=N_TRIALS, epochs=EPOCHS,
        eta=ETA, batch_size=BATCH_SIZE)

    plot_ablation(results, epochs=EPOCHS, checkpoint_every=CHECKPOINT_EVERY)

    # Summary table
    print(f"\n{'Config':<10} {'Train(best)':>12} {'Train(final)':>14} "
          f"{'Test(best)':>12} {'Test(final)':>13} {'Std(test)':>11}")
    print("-" * 65)
    for cfg_key in ['A', 'B', 'C', 'D']:
        test_hists  = results[cfg_key]['hist']        # test acc history
        train_hists = results[cfg_key]['train_hist']  # train acc history
        best_test   = np.array([max(h) for h in test_hists])
        final_test  = np.array([h[-1] for h in test_hists])
        best_train  = np.array([max(h) for h in train_hists])
        final_train = np.array([h[-1] for h in train_hists])
        print(f"  {cfg_key:<8} {best_train.mean():>12.1%} {final_train.mean():>14.1%} "
              f"{best_test.mean():>12.1%} {final_test.mean():>13.1%} "
              f"{final_test.std():>11.1%}")