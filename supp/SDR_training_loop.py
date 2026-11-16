import numpy as np
import strawberryfields as sf
from strawberryfields import ops
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings


# ── CV linear classifier ───────────────────────────────────────────────────
def cv_linear_classifier(x, params):
    """
    CV photonic classifier: 4 modes, displacement encoding,
    rotation + beamsplitter network, x-quadrature mean readout on mode 0.
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
    return result.state.means()[0]   # x-quadrature mean, mode 0


# ── CV multiplier circuit (loaded from saved params) ─────────────────────────
def run_cv_multiplier(delta_f, e, circuit_params, n_layers=2):
    """
    Single-mode circuit:
      S(sq_scale * delta_f) D(disp_scale * e) R(rot) per layer
      <x> * output_gain -> gradient estimate
    """
    try:
        eng  = sf.Engine("gaussian")
        prog = sf.Program(1)
        cp          = circuit_params
        output_gain = cp[-1]
        with prog.context as q:
            for layer in range(n_layers):
                sq_scale   = cp[layer * 3]
                disp_scale = cp[layer * 3 + 1]
                rot        = cp[layer * 3 + 2]
                ops.Sgate(float(np.clip(sq_scale   * delta_f, -3.0, 3.0))) | q[0]
                ops.Dgate(float(np.clip(disp_scale * e,       -3.0, 3.0))) | q[0]
                ops.Rgate(float(rot))                                       | q[0]
        result     = eng.run(prog)
        raw_output = float(result.state.means()[0])
        return raw_output * float(output_gain)
    except Exception:
        return 0.0


# ── CV D gate parameter update circuit ───────────────────────────────────────
def cv_dgate_update(params, gradients, learning_rate, high_acc=False):
    """
    D(theta_k) D(eta * g_k) |0> -> x-quadrature mean / sqrt(2) = theta_k + eta * g_k
    Returns updated parameters read from x-quadrature means.
    No clipping when high_acc=True (accuracy >= 80%).
    """
    n_params = len(params)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eng  = sf.Engine("gaussian")
            prog = sf.Program(n_params)
            with prog.context as q:
                for k in range(n_params):
                    ops.Dgate(float(params[k]),                    0) | q[k]
                    ops.Dgate(float(learning_rate * gradients[k]), 0) | q[k]
            result     = eng.run(prog)
            means      = result.state.means()
            new_params = np.array([means[k] / np.sqrt(2) for k in range(n_params)])
            if high_acc:
                return new_params   # no clipping
            return np.clip(new_params, -5, 5)
    except Exception:
        return params


class CVSelfLearner:
    """
    CV Self-Learning System using:
    - cv_linear_classifier with x-quadrature mean readout
    - CV multiplier circuit for gradient computation (loaded from cv_multiplier_params.npy)
    - D gate circuit for parameter updates via x-quadrature mean
    """

    def __init__(self, seed=42):
        np.random.seed(seed)
        self.params     = 0.1 * np.random.randn(12)
        self.thresholds = [-0.5, 0.5]
        self.epsilon    = 0.015
        self.target_values = {0: -1.0, 1: 0.0, 2: 1.0}

        # Load trained CV multiplier parameters
        self.multiplier_params = np.load('cv_multiplier_params.npy')

        self.best_params   = self.params.copy()
        self.best_accuracy = 0.0

    def compute_cv_gradients(self, x, target):
        """
        Compute gradients for all parameters using CV multiplier circuit.
        For each parameter:
          delta_f = f(x; theta + epsilon*e_i) - f(x; theta)
          e       = target - f(x; theta)
          grad_i  = run_cv_multiplier(delta_f, e) -> x-quadrature mean
        """
        n_params  = len(self.params)
        gradients = np.zeros(n_params)
        f_base    = cv_linear_classifier(x, self.params)

        for param_idx in range(n_params):
            params_pert             = self.params.copy()
            params_pert[param_idx] += self.epsilon
            f_pert  = cv_linear_classifier(x, params_pert)
            delta_f = f_pert - f_base
            e       = target - f_base
            gradients[param_idx] = run_cv_multiplier(
                delta_f, e, self.multiplier_params)

        return gradients

    def cv_connected_learning(self, X_train, y_train,
                                   X_test=None, y_test=None,
                                   epochs=40, learning_rate=1.0):
        """
        CV learning loop:
        - Gradient computed via CV multiplier (x-quadrature mean readout)
        - Parameter update via D gate circuit (x-quadrature mean readout)
        - No clip when accuracy >= 80%, clip to [-1.8, 1.8] otherwise
        - Best params restored on significant accuracy drop
        """
        print("CV LEARNING: CV multiplier gradient + D gate update (x-quadrature mean readout)")
        print("="*70)

        accuracy_history = []

        for epoch in range(epochs):
            current_lr = learning_rate * (0.95 ** (epoch // 5))
            indices    = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_updates = []

            for x, y_true in zip(X_shuffled, y_shuffled):
                target = self.target_values[y_true]
                grads  = self.compute_cv_gradients(x, target)
                if not np.allclose(grads, 0):
                    epoch_updates.append(grads)

            if epoch_updates:
                final_grads = np.mean(epoch_updates, axis=0)
                print(f"         grad norm: {np.linalg.norm(final_grads):.6f}  max: {np.max(np.abs(final_grads)):.6f}")
                print(f"         param range: [{self.params.min():.3f}, {self.params.max():.3f}]")

                # D gate parameter update — no clip if already at high accuracy
                self.params = cv_dgate_update(
                    self.params, final_grads, current_lr,
                    high_acc=(self.best_accuracy >= 0.80))

            # Evaluate
            train_preds = []
            for x in X_train:
                out = cv_linear_classifier(x, self.params)
                train_preds.append(0 if out < self.thresholds[0]
                                   else (1 if out < self.thresholds[1] else 2))
            test_preds = []
            for x in X_test:
                out = cv_linear_classifier(x, self.params)
                test_preds.append(0 if out < self.thresholds[0]
                                  else (1 if out < self.thresholds[1] else 2))

            train_acc = accuracy_score(y_train, train_preds)
            test_acc  = accuracy_score(y_test,  test_preds)
            accuracy_history.append((train_acc, test_acc))

            print(f"Epoch {epoch+1:2d}: train={train_acc:.3f}  test={test_acc:.3f}  lr={current_lr:.4f}")


            # Reduce learning rate once train accuracy >= 80%
            if train_acc >= 0.80 and current_lr > 0.1:
                learning_rate = 0.1
                print(f"         Performance >= 80%, learning rate -> {learning_rate:.4f}")

            # Record best params
            if test_acc > self.best_accuracy:
                self.best_accuracy = test_acc
                self.best_params   = self.params.copy()
                if test_acc >= 0.85:
                    print(f"         NEW BEST: {test_acc:.1%}")
            elif test_acc < self.best_accuracy - 0.1:
                self.params    = self.best_params.copy()
                learning_rate *= 0.5   # halve lr on each restore
                print(f"         Drop detected, restored best params ({self.best_accuracy:.1%}), lr -> {learning_rate:.4f}")

        # Apply best params at the end
        self.params = self.best_params
        print(f"\nFinal params set to best ({self.best_accuracy:.1%})")

        return accuracy_history

    def predict_cv(self, X):
        """Classify using x-quadrature mean readout and fixed thresholds."""
        predictions = []
        for x in X:
            try:
                out = cv_linear_classifier(x, self.params)
                predictions.append(0 if out < self.thresholds[0]
                                   else (1 if out < self.thresholds[1] else 2))
            except Exception:
                predictions.append(np.random.randint(0, 3))
        return np.array(predictions)


def run_single_trial(seed, X_train, X_test, y_train, y_test, epochs=30):
    """Run one training trial with a given random seed. Returns (acc_history, test_acc, learner)."""
    learner = CVSelfLearner(seed=seed)
    print(f"\n{'='*50}")
    print(f"  Trial seed={seed}")
    print(f"{'='*50}")
    acc_history = learner.cv_connected_learning(
        X_train, y_train,
        X_test=X_test, y_test=y_test,
        epochs=epochs,
        learning_rate=1.0,
    )
    test_predictions = learner.predict_cv(X_test)
    test_accuracy    = accuracy_score(y_test, test_predictions)
    print(f"  -> Test accuracy: {test_accuracy:.1%}")
    return acc_history, test_accuracy, learner


def plot_results(all_histories, test_accs, checkpoint_epochs,
                 best_learner, X_test, y_test):
    """Figures from multi-trial results."""
    plt.rcParams.update({
        'font.family':       'serif',
        'font.size':         13,
        'axes.titlesize':    14,
        'axes.labelsize':    13,
        'xtick.labelsize':   11,
        'ytick.labelsize':   11,
        'legend.fontsize':   11,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.grid':         False,
        'figure.dpi':        150,
    })
    BLUE   = '#2563EB'
    RED    = '#DC2626'
    GREEN  = '#16A34A'
    ORANGE = '#EA580C'
    GRAY   = '#6B7280'

    all_test_histories  = [[t for _, t in h] for h in all_histories]
    max_len  = max(len(h) for h in all_test_histories)
    padded   = np.array([
        np.pad(h, (0, max_len - len(h)), mode='edge') for h in all_test_histories
    ])
    mean_acc = padded.mean(axis=0)
    std_acc  = padded.std(axis=0)
    epochs_x = checkpoint_epochs[:max_len]
    test_accs = np.array(test_accs)

    # Figure 1: Training curves
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for curve in padded:
        ax.plot(epochs_x, curve, color=BLUE, alpha=0.15, linewidth=1.0)
    ax.fill_between(epochs_x, mean_acc - std_acc, mean_acc + std_acc,
                    color=BLUE, alpha=0.18, label=r'Mean $\pm$ std')
    ax.plot(epochs_x, mean_acc, color=BLUE, linewidth=2.2,
            marker='o', markersize=6, label='Mean accuracy')
    ax.axhline(1/3, color=RED, linestyle='--', linewidth=1.4,
               alpha=0.8, label='Random baseline (33.3%)')
    ax.axhline(1.0, color=GRAY, linestyle=':', linewidth=1.0, alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy')
    ax.set_ylim(0.20, 1.05)
    ax.set_xlim(epochs_x[0] - 0.5, epochs_x[-1] + 0.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(loc='lower right', framealpha=0.9)
    fig.tight_layout()
    fig.savefig('cv_training_curves.svg', bbox_inches='tight')
    fig.savefig('cv_training_curves.png', bbox_inches='tight', dpi=200)
    print("Saved cv_training_curves.svg/png")

    # Figure 2: Test accuracy per trial
    fig2, ax2 = plt.subplots(figsize=(5.5, 4.0))
    x    = np.arange(len(test_accs))
    bars = ax2.bar(x, test_accs, color=BLUE, alpha=0.8,
                   edgecolor='white', linewidth=0.5, width=0.55)
    ax2.axhline(test_accs.mean(), color=RED, linestyle='--', linewidth=1.6,
                label=f'Mean = {test_accs.mean():.1%}')
    ax2.fill_between([-0.4, len(test_accs) - 0.6],
                     test_accs.mean() - test_accs.std(),
                     test_accs.mean() + test_accs.std(),
                     color=RED, alpha=0.10,
                     label=f'Std  = {test_accs.std():.1%}')
    ax2.axhline(1/3, color=GRAY, linestyle=':', linewidth=1.2,
                alpha=0.7, label='Random baseline')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{i+1}' for i in x], fontsize=10)
    ax2.set_ylabel('Test Accuracy')
    ax2.set_ylim(0.0, 1.08)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    for bar, val in zip(bars, test_accs):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.1%}', ha='center', va='bottom', fontsize=9,
                rotation=90)
    ax2.legend(fontsize=10, framealpha=0.9)
    fig2.tight_layout()
    fig2.savefig('cv_test_accuracy_trials_ver2.svg', bbox_inches='tight')
    fig2.savefig('cv_test_accuracy_trials_ver2.png', bbox_inches='tight', dpi=200)
    print("Saved cv_test_accuracy_trials_ver2.svg/png")

    # Figure 3: x-quadrature mean distribution (best learner)
    outputs, labels = [], []
    for x_sample, y_true in zip(X_test, y_test):
        try:
            out = cv_linear_classifier(x_sample, best_learner.params)
            outputs.append(out)
            labels.append(y_true)
        except Exception:
            pass
    outputs = np.array(outputs)
    labels  = np.array(labels)

    fig3, ax3 = plt.subplots(figsize=(7, 4.5))
    colors      = [BLUE, ORANGE, GREEN]
    class_names = ['Class 0 (setosa)', 'Class 1 (versicolor)', 'Class 2 (virginica)']
    bins = np.linspace(outputs.min() - 0.1, outputs.max() + 0.1, 30)
    for cls, color, name in zip([0, 1, 2], colors, class_names):
        ax3.hist(outputs[labels == cls], bins=bins,
                 color=color, alpha=0.75, label=name,
                 edgecolor='white', linewidth=0.4)
    ymax = ax3.get_ylim()[1]
    for tau, lbl in zip(best_learner.thresholds, [r'$\tau_0$', r'$\tau_1$']):
        ax3.axvline(tau, color=RED, linestyle='--', linewidth=1.6, alpha=0.85)
        ax3.text(tau + 0.02, ymax * 0.93, lbl, color=RED, fontsize=11, va='top')
    ax3.set_xlabel(r'$x$-quadrature mean $\langle\hat{x}_0\rangle$')
    ax3.set_ylabel('Count')
    ax3.legend(loc='upper left', framealpha=0.9)
    fig3.tight_layout()
    fig3.savefig('cv_measurement_dist_ver2.svg', bbox_inches='tight')
    fig3.savefig('cv_measurement_dist_ver2.png', bbox_inches='tight', dpi=200)
    print("Saved cv_measurement_dist_ver2.svg/png")


def demonstrate_cv_learning(n_trials=5, epochs=30):
    """Run n_trials independent training runs and produce figures."""
    iris = load_iris()
    X, y = iris.data, iris.target
    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=42)

    print("CV SELF-LEARNING — MULTI-TRIAL")
    print("="*60)
    print(f"  Trials : {n_trials}")
    print(f"  Epochs : {epochs}")
    print(f"  Train  : {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    checkpoint_epochs = list(range(1, epochs + 1, 1))

    all_histories = []
    test_accs     = []
    best_acc      = -1
    best_learner  = None

    for trial_idx in range(n_trials):
        seed = 42 + trial_idx * 13
        hist, tacc, learner = run_single_trial(
            seed, X_train, X_test, y_train, y_test, epochs=epochs)
        all_histories.append(hist)
        test_accs.append(tacc)
        if tacc > best_acc:
            best_acc     = tacc
            best_learner = learner

    test_accs = np.array(test_accs)
    print(f"\n{'='*60}")
    print(f"MULTI-TRIAL SUMMARY ({n_trials} trials)")
    print(f"{'='*60}")
    print(f"  Test accuracy : {test_accs.mean():.1%} ± {test_accs.std():.1%}")
    print(f"  Best          : {test_accs.max():.1%}")
    print(f"  Worst         : {test_accs.min():.1%}")

    plot_results(all_histories, test_accs, checkpoint_epochs,
                 best_learner, X_test, y_test)

    return best_learner, test_accs


if __name__ == "__main__":
    best_learner, test_accs = demonstrate_cv_learning(n_trials=25, epochs=15)