"""
Trainable CV Multiplier Circuit
================================
Uses squeezing-displacement encoding for analog multiplication.
Target: classical_grad = (delta_f / epsilon) * e
Loss: 1 - R^2 (scale-invariant)
Dataset: real Iris samples x random theta sweeps.
Trained parameters saved to cv_multiplier_params.npy
"""

import numpy as np
import strawberryfields as sf
from strawberryfields import ops
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

EPSILON    = 0.015
N_MODES    = 1    # single mode
N_LAYERS   = 3
TARGET_MAP = {0: -1.0, 1: 0.0, 2: 1.0}


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
    result = eng.run(prog)
    return result.state.means()[0]

def generate_multiplier_dataset(X, y, n_theta=50, n_params=12):
    records = []
    for x_sample, y_true in zip(X, y):
        target = TARGET_MAP[y_true]
        for _ in range(n_theta):
            params    = np.random.uniform(-0.3, 0.3, size=12)
            param_idx = np.random.randint(n_params)

            f_base = cv_linear_classifier(x_sample, params)
            params[param_idx] += EPSILON
            f_pert = cv_linear_classifier(x_sample, params)
            params[param_idx] -= EPSILON

            delta_f        = f_pert - f_base
            e              = target - f_base
            classical_grad = (delta_f / EPSILON) * e

            records.append({
                'delta_f':        delta_f,
                'e':              e,
                'classical_grad': classical_grad,
            })
    return records


def compute_stats(dataset):
    delta_f_std = np.std([r['delta_f'] for r in dataset]) + 1e-8
    e_std       = np.std([r['e']       for r in dataset]) + 1e-8
    return delta_f_std, e_std


def apply_normalization(dataset, delta_f_std, e_std):
    normalized = []
    for r in dataset:
        normalized.append({
            'delta_f':        r['delta_f'] / delta_f_std,
            'e':              r['e']       / e_std,
            'classical_grad': r['classical_grad'],
        })
    return normalized


def n_circuit_params(n_layers):
    # per layer: sq_scale, disp_scale, rot
    return n_layers * 3 


def run_cv_multiplier(delta_f, e, circuit_params, n_layers=2):
    """
    Single-mode circuit:
      S(sq_scale * delta_f) D(disp_scale * e) R(rot) per layer
      x-quadrature mean readout
    """
    try:
        eng  = sf.Engine("gaussian")
        prog = sf.Program(1)
        cp   = circuit_params

        with prog.context as q:
            for layer in range(n_layers):
                sq_scale   = cp[layer * 3]
                disp_scale = cp[layer * 3 + 1]
                rot        = cp[layer * 3 + 2]
                ops.Dgate(float(np.clip(disp_scale * e,       -3.0, 3.0))) | q[0]
                ops.Sgate(float(np.clip(sq_scale   * delta_f, -3.0, 3.0))) | q[0]
                ops.Rgate(float(rot))                                       | q[0]

        result = eng.run(prog)
        return float(result.state.means()[0])   # x-quadrature mean
    except Exception:
        return 0.0

def compute_loss(circuit_params, dataset, n_layers=2):
    """1 - R^2 loss: scale-invariant."""
    outputs, targets = [], []
    for rec in dataset:
        out = run_cv_multiplier(rec['delta_f'], rec['e'],
                                circuit_params, n_layers)
        outputs.append(out)
        targets.append(rec['classical_grad'])
    outputs = np.array(outputs)
    targets = np.array(targets)
    var_t   = np.var(targets) + 1e-10
    r2      = 1.0 - np.mean((outputs - targets)**2) / var_t
    return 1.0 - r2


def train_multiplier(dataset_train, dataset_val,
                     n_layers=2, n_iter=400, lr=0.1, decay=0.997):
    n_params  = n_circuit_params(n_layers)
    cp = np.random.randn(n_params) * 0.05
    cp[1::3] = 1.0   # initialize all disp_scale params to 1.0
    best_cp   = cp.copy()
    best_loss = compute_loss(cp, dataset_train, n_layers)
    train_losses, val_losses = [], []
    print(f"Initial train loss (1-R^2) : {best_loss:.6f}")
    print(f"n_circuit_params           : {n_params}")
    print(f"Training samples           : {len(dataset_train)}")
    print(f"Validation samples         : {len(dataset_val)}")
    print()
    current_lr = lr
    for it in range(n_iter):
        idx          = np.random.randint(n_params)
        delta        = np.random.choice([-1, 1]) * current_lr
        cp_new       = cp.copy()
        cp_new[idx] += delta
        loss_new     = compute_loss(cp_new, dataset_train, n_layers)
        if loss_new < best_loss:
            best_loss = loss_new
            best_cp   = cp_new.copy()
            cp        = cp_new.copy()
        current_lr *= decay
        if (it + 1) % 50 == 0:
            val_loss = compute_loss(best_cp, dataset_val, n_layers)
            train_losses.append(best_loss)
            val_losses.append(val_loss)
            print(f"Iter {it+1:4d}: train 1-R^2 = {best_loss:.6f}  |  val 1-R^2 = {val_loss:.6f}  |  lr = {current_lr:.5f}")
    return best_cp, train_losses, val_losses


def evaluate_multiplier(circuit_params, dataset, n_layers=2, label=''):
    outputs, targets = [], []
    for rec in dataset:
        out = run_cv_multiplier(rec['delta_f'], rec['e'],
                                circuit_params, n_layers)
        outputs.append(out)
        targets.append(rec['classical_grad'])
    outputs = np.array(outputs)
    targets = np.array(targets)
    var_t   = np.var(targets) + 1e-10
    r2      = 1.0 - np.mean((outputs - targets)**2) / var_t
    mae     = np.mean(np.abs(outputs - targets))
    corr    = np.corrcoef(outputs, targets)[0, 1] if np.std(outputs) > 1e-10 else 0.0
    print(f"\n{label} Evaluation ({len(dataset)} samples):")
    print(f"  R^2         : {r2:.4f}")
    print(f"  MAE         : {mae:.6f}")
    print(f"  Pearson r   : {corr:.4f}")
    print(f"  Output range: [{outputs.min():.4f}, {outputs.max():.4f}]")
    print(f"  Target range: [{targets.min():.4f}, {targets.max():.4f}]")
    return outputs, targets, r2, mae, corr


def plot_results(train_losses, val_losses, outputs_tr, targets_tr, outputs_val, targets_val):
    plt.rcParams.update({
        'font.family':        'serif',
        'font.size':          9,
        'axes.labelsize':     10,
        'legend.fontsize':    8,
        'xtick.labelsize':    8,
        'ytick.labelsize':    8,
        'axes.linewidth':     0.8,
        'xtick.major.width':  0.8,
        'ytick.major.width':  0.8,
        'xtick.direction':    'in',
        'ytick.direction':    'in',
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'savefig.dpi':        300,
        'savefig.bbox':       'tight',
        'savefig.pad_inches': 0.05,
    })
    BLUE = '#2563EB'; RED = '#DC2626'; TEAL = '#2A9D8F'
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    fig.subplots_adjust(wspace=0.38)

    ax    = axes[0]
    iters = np.arange(50, 50 * len(train_losses) + 1, 50)
    ax.plot(iters, train_losses, c=BLUE, lw=1.5, marker='o', ms=4, label=r'Train $1-R^2$')
    ax.plot(iters, val_losses,   c=RED,  lw=1.5, marker='s', ms=4, label=r'Val $1-R^2$')
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'$1 - R^2$')
    ax.set_title('(a) Training loss', loc='left', pad=4)
    ax.legend(frameon=False)

    ax  = axes[1]
    lim = max(np.abs(targets_tr).max(), np.abs(outputs_tr).max()) * 1.1
    ax.scatter(targets_tr, outputs_tr, c=BLUE, s=8, alpha=0.5, linewidths=0)
    ax.plot([-lim, lim], [-lim, lim], c=RED, lw=0.9, ls='--', label='Ideal')
    ax.set_xlabel(r'Classical gradient $g_\mathrm{classical}$')
    ax.set_ylabel(r'Circuit output $\langle\hat{x}_0\rangle$')
    ax.set_title('(b) Train set', loc='left', pad=4)
    corr_tr = np.corrcoef(targets_tr, outputs_tr)[0, 1]
    ax.text(0.97, 0.04, f'$r = {corr_tr:.3f}$',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#cccccc', lw=0.6))
    ax.legend(frameon=False)

    ax  = axes[2]
    lim = max(np.abs(targets_val).max(), np.abs(outputs_val).max()) * 1.1
    ax.scatter(targets_val, outputs_val, c=TEAL, s=8, alpha=0.6, linewidths=0)
    ax.plot([-lim, lim], [-lim, lim], c=RED, lw=0.9, ls='--', label='Ideal')
    ax.set_xlabel(r'Classical gradient $g_\mathrm{classical}$')
    ax.set_ylabel(r'Circuit output $\langle\hat{x}_0\rangle$')
    ax.set_title('(c) Validation set', loc='left', pad=4)
    corr_val = np.corrcoef(targets_val, outputs_val)[0, 1]
    ax.text(0.97, 0.04, f'$r = {corr_val:.3f}$',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#cccccc', lw=0.6))
    ax.legend(frameon=False)

    fig.savefig('cv_gradient_train.png')
    fig.savefig('cv_gradient_train.pdf')
    print("Saved cv_gradient_train.png/pdf")


if __name__ == '__main__':
    iris = load_iris()
    X, y = iris.data, iris.target
    X    = MinMaxScaler(feature_range=(-0.8, 0.8)).fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=42)

    print("Generating multiplier dataset...")
    dataset_all = generate_multiplier_dataset(X_train, y_train, n_theta=50)
    split       = int(0.8 * len(dataset_all))
    train_raw   = dataset_all[:split]
    val_raw     = dataset_all[split:]

    # Normalize inputs by train stats only
    delta_f_std, e_std = compute_stats(train_raw)
    dataset_train      = apply_normalization(train_raw, delta_f_std, e_std)
    dataset_val_s      = apply_normalization(val_raw,   delta_f_std, e_std)

    print(f"Train records : {len(dataset_train)}")
    print(f"Val   records : {len(dataset_val_s)}")

    print("\n" + "="*55)
    print("TRAINING TRAINABLE CV MULTIPLIER")
    print("="*55)
    best_cp, train_losses, val_losses = train_multiplier(
        dataset_train, dataset_val_s,
        n_layers=N_LAYERS,
        n_iter=400, lr=0.1, decay=0.997,
    )

    # Save trained circuit parameters
    np.save('cv_multiplier_params.npy', best_cp)
    print(f"\nSaved circuit parameters to cv_multiplier_params.npy")
    print(f"Circuit params: {best_cp}")

    out_tr,  tgt_tr,  r2_tr,  mae_tr,  corr_tr  = evaluate_multiplier(
        best_cp, dataset_train, label='Train')
    out_val, tgt_val, r2_val, mae_val, corr_val = evaluate_multiplier(
        best_cp, dataset_val_s, label='Validation')

    plot_results(train_losses, val_losses, out_tr, tgt_tr, out_val, tgt_val)

    print("\nDone.")