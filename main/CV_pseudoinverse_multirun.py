import numpy as np
import strawberryfields as sf
from strawberryfields import ops
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load and normalize Iris data
iris = load_iris()
X, y = iris.data, iris.target
X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)  # Scale for displacement ops

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)

# Linear circuit for classification
def linear_classifier_circuit(x, params):
    """
    Fully linear classifier circuit that directly outputs class via measurement
    
    Args:
        x: normalized input features (4 features)
        params: circuit parameters (12 parameters)
        
    Returns:
        measurement result that directly indicates class
    """
    # Create engine with Gaussian backend
    eng = sf.Engine("gaussian")
    
    # Use 4 modes - one for each feature
    prog = sf.Program(4)
    
    # Unpack parameters
    rotation_angles = params[:4]  # 4 rotation parameters
    bs_params = params[4:12]  # 8 beam splitter parameters (4 pairs)
    
    with prog.context as q:
        # Input encoding via displacement
        for i in range(4):
            ops.Dgate(x[i], 0) | q[i]
        
        # Apply rotation gates to each mode
        for i in range(4):
            ops.Rgate(rotation_angles[i]) | q[i]
        
        # First layer of beam splitters
        ops.BSgate(bs_params[0], bs_params[1]) | (q[0], q[1])
        ops.BSgate(bs_params[2], bs_params[3]) | (q[2], q[3])
        
        # Second layer of beam splitters
        ops.BSgate(bs_params[4], bs_params[5]) | (q[0], q[2])
        ops.BSgate(bs_params[6], bs_params[7]) | (q[1], q[3])
        
        # Final concentration to single mode (mode 0)
        ops.BSgate(np.pi/4, 0) | (q[0], q[1])
        ops.BSgate(np.pi/4, 0) | (q[0], q[3])
    
    # Run circuit
    result = eng.run(prog)
    
    # Get X quadrature of the output mode (mode 0)
    means = result.state.means()
    output_value = means[0]  # X quadrature of first mode
    
    return output_value

# Function to convert measurement to class prediction
def measurement_to_class(measurement, thresholds):
    """
    Convert continuous measurement to discrete class
    
    Args:
        measurement: linear measurement outcome
        thresholds: list of two threshold values [t1, t2]
        
    Returns:
        predicted class (0, 1, or 2)
    """
    if measurement < thresholds[0]:
        return 0
    elif measurement < thresholds[1]:
        return 1
    else:
        return 2

# Prediction function
def predict(X, params, thresholds):
    """
    Predict classes for multiple samples
    
    Args:
        X: input features
        params: circuit parameters
        thresholds: classification thresholds
        
    Returns:
        array of predicted classes
    """
    predictions = []
    for x in X:
        try:
            # Get linear measurement
            measurement = linear_classifier_circuit(x, params)
            # Convert to class
            pred_class = measurement_to_class(measurement, thresholds)
            predictions.append(pred_class)
        except Exception as error:
            print(f"Circuit error: {error}")
            # Fallback to random prediction if circuit fails
            predictions.append(np.random.randint(0, 3))
    
    return np.array(predictions)

# Calculate Jacobian for pseudoinverse
def compute_jacobian(x, target_value, params, epsilon=0.05):
    """
    Compute Jacobian of the measurement with respect to parameters
    
    Args:
        x: input features
        target_value: target measurement value for this class
        params: current parameters
        epsilon: small value for finite difference
        
    Returns:
        Jacobian vector
    """
    try:
        # Base measurement
        base_measurement = linear_classifier_circuit(x, params)
        jacobian = []
        
        # Compute partial derivatives for each parameter
        for i in range(len(params)):
            params_perturbed = params.copy()
            params_perturbed[i] += epsilon
            
            # Measure with perturbed parameter
            perturbed_measurement = linear_classifier_circuit(x, params_perturbed)
            
            # Finite difference approximation
            derivative = (perturbed_measurement - base_measurement) / epsilon
            jacobian.append(derivative)
        
        return np.array(jacobian), base_measurement
    except Exception as err:
        print(f"Jacobian computation error: {err}")
        return None, None

# Pseudoinverse training
def train_linear_classifier(X_train, y_train, epochs=20, eta=0.1, batch_size=5):
    """
    Train linear classifier using pseudoinverse method
    
    Args:
        X_train: training features
        y_train: training labels
        epochs: number of training epochs
        eta: learning rate
        batch_size: mini-batch size
        
    Returns:
        trained parameters and thresholds
    """
    # Target values for each class (X quadrature targets)
    target_values = {
        0: -1.0,  # Target for class 0
        1: 0.0,   # Target for class 1
        2: 1.0    # Target for class 2
    }
    
    # Initialize parameters (12 circuit parameters)
    #np.random.seed(42)
    params = 0.1 * np.random.randn(12)
    
    # Initialize thresholds
    thresholds = [-0.5, 0.5]  # Initial thresholds between classes
    
    acc_history = []
    
    # Update function using pseudoinverse
    def update_parameters(params, X_batch, y_batch, eta):
        jacobians = []
        delta_measurements = []
        actual_measurements = []
        
        for x, y_true in zip(X_batch, y_batch):
            # Get target measurement for this class
            target_measurement = target_values[y_true]
            
            # Compute Jacobian and current measurement
            J, current_measurement = compute_jacobian(x, target_measurement, params)
            
            if J is not None:
                jacobians.append(J)
                delta_measurements.append(target_measurement - current_measurement)
                actual_measurements.append(current_measurement)
        
        if not jacobians:
            return params
        
        # Stack Jacobians and deltas
        J_matrix = np.stack(jacobians)
        delta_m_vector = np.array(delta_measurements).reshape(-1, 1)
        
        try:
            # Add small regularization for stability
            reg_factor = 0.01
            J_transpose = J_matrix.T
            regularized_inverse = np.linalg.inv(J_transpose @ J_matrix + 
                                               reg_factor * np.eye(J_transpose.shape[0]))
            
            # Compute pseudoinverse update
            delta_params = regularized_inverse @ J_transpose @ delta_m_vector
            new_params = params + eta * delta_params.flatten()
            
            # Also update thresholds based on actual measurements
            if len(actual_measurements) > 0:
                actual_measurements = np.array(actual_measurements)
                actual_classes = np.array(y_batch)
                
                # Group measurements by class
                measurements_by_class = [
                    actual_measurements[actual_classes == c] for c in range(3)
                ]
                
                # Update thresholds if we have measurements for adjacent classes
                # Damping factor: thresholds move slowly toward new midpoint
                threshold_damping = 0.1
                if len(measurements_by_class[0]) > 0 and len(measurements_by_class[1]) > 0:
                    new_tau0 = (np.mean(measurements_by_class[0]) +
                                np.mean(measurements_by_class[1])) / 2
                    thresholds[0] = (1 - threshold_damping) * thresholds[0] + threshold_damping * new_tau0

                if len(measurements_by_class[1]) > 0 and len(measurements_by_class[2]) > 0:
                    new_tau1 = (np.mean(measurements_by_class[1]) +
                                np.mean(measurements_by_class[2])) / 2
                    thresholds[1] = (1 - threshold_damping) * thresholds[1] + threshold_damping * new_tau1
            
            return new_params
            
        except np.linalg.LinAlgError:
            print("Pseudoinverse calculation failed, using original parameters")
            return params
    
    # Training loop
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        # Mini-batch updates
        for i in range(0, len(X_train), batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            if len(X_batch) == batch_size:
                params = update_parameters(params, X_batch, y_batch, eta)
        
        # Evaluate on a subset
        if (epoch + 1) % 1 == 0:
            subset_size = min(20, len(X_train))
            subset_idx = np.random.choice(len(X_train), subset_size, replace=False)
            X_subset = X_train[subset_idx]
            y_subset = y_train[subset_idx]
            
            predictions = predict(X_subset, params, thresholds)
            accuracy = accuracy_score(y_subset, predictions)
            acc_history.append(accuracy)
            
            print(f"Epoch {epoch+1}: Subset accuracy = {accuracy:.4f}")
            print(f"Current thresholds: {thresholds}")
    
    return params, thresholds, acc_history

def run_single_trial(seed, X_train, X_test, y_train, y_test,
                     epochs=20, eta=0.1, batch_size=5):
    """Run one training trial with a given random seed.
    Returns (acc_history, test_acc, params, thresholds).
    """
    np.random.seed(seed)
    print(f"\n{'='*50}")
    print(f"  Trial  seed={seed}")
    print(f"{'='*50}")
    params, thresholds, acc_history = train_linear_classifier(
        X_train, y_train, epochs=epochs, eta=eta, batch_size=batch_size
    )
    test_preds = predict(X_test, params, thresholds)
    test_acc   = accuracy_score(y_test, test_preds)
    print(f"  -> Test accuracy: {test_acc:.1%}")
    return acc_history, test_acc, params, thresholds


def plot_results(all_histories, test_accs, epochs, checkpoint_every,
                 best_params, best_thresholds, X_test, y_test):
    """Figures from multi-trial results."""

    # ── Style ─────────────────────────────────────────────────
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
        'axes.grid':         True,
        'grid.alpha':        0.25,
        'grid.linestyle':    '--',
        'figure.dpi':        150,
    })
    BLUE   = '#2563EB'
    RED    = '#DC2626'
    GREEN  = '#16A34A'
    ORANGE = '#EA580C'
    GRAY   = '#6B7280'

    # Epoch checkpoints (every checkpoint_every epochs)
    checkpoint_epochs = list(range(checkpoint_every,
                                   epochs + 1,
                                   checkpoint_every))

    # Align histories (pad with last value if runs differ in length)
    max_len = max(len(h) for h in all_histories)
    padded  = np.array([
        np.pad(h, (0, max_len - len(h)), mode='edge')
        for h in all_histories
    ])
    epochs_x = np.array(checkpoint_epochs[:max_len])
    mean_acc = padded.mean(axis=0)
    std_acc  = padded.std(axis=0)
    test_accs = np.array(test_accs)

    # ── Figure 1: Training curves ──────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for curve in padded:
        ax.plot(epochs_x, curve, color=BLUE, alpha=0.15, linewidth=1.0)
    ax.fill_between(epochs_x,
                    mean_acc - std_acc,
                    mean_acc + std_acc,
                    color=BLUE, alpha=0.18, label=r'Mean $\pm$ std')
    ax.plot(epochs_x, mean_acc, color=BLUE, linewidth=2.2,
            marker='o', markersize=6, label='Mean accuracy')
    ax.axhline(1/3, color=RED,  linestyle='--', linewidth=1.4,
               alpha=0.8, label='Random baseline (33.3%)')
    ax.axhline(1.0, color=GRAY, linestyle=':',  linewidth=1.0, alpha=0.4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Accuracy')
    ax.set_title(f'CV linear Classifier — Training Accuracy\n'
                 f'({len(padded)} independent trials, Iris 3-class, pseudoinverse)')
    ax.set_ylim(0.20, 1.08)
    ax.set_xlim(epochs_x[0] - 0.5, epochs_x[-1] + 0.5)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(False)   # Figure 1
    fig.tight_layout()
    fig.savefig('cv_pseudoinverse_training_curves.svg', bbox_inches='tight')
    fig.savefig('cv_pseudoinverse_training_curves.png',
                bbox_inches='tight', dpi=200)
    print("Saved cv_pseudoinverse_training_curves.svg/png")

    # ── Figure 2: Test accuracy per trial ─────────────────────
    fig2, ax2 = plt.subplots(figsize=(5.5, 4.0))
    x = np.arange(len(test_accs))
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
    ax2.set_xticklabels([f'Trial {i+1}' for i in x], fontsize=10)
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('Test Accuracy Across Independent Trials\n(pseudoinverse method)')
    ax2.set_ylim(0.0, 1.08)
    ax2.grid(False)  # Figure 2
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    for bar, val in zip(bars, test_accs):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f'{val:.1%}', ha='center', va='bottom', fontsize=9)
    ax2.legend(fontsize=10, framealpha=0.9)
    fig2.tight_layout()
    fig2.savefig('cv_pseudoinverse_test_accuracy.svg', bbox_inches='tight')
    fig2.savefig('cv_pseudoinverse_test_accuracy.png',
                 bbox_inches='tight', dpi=200)
    print("Saved cv_pseudoinverse_test_accuracy.svg/png")

    # ── Figure 3: Measurement distribution (best trial) ───────
    measurements, labels = [], []
    for x_sample, y_true in zip(X_test, y_test):
        try:
            m = linear_classifier_circuit(x_sample, best_params)
            measurements.append(m)
            labels.append(y_true)
        except Exception:
            pass
    measurements = np.array(measurements)
    labels       = np.array(labels)

    fig3, ax3 = plt.subplots(figsize=(6, 4.0))
    colors      = [BLUE, ORANGE, GREEN]
    class_names = ['Class 0 (setosa)', 'Class 1 (versicolor)', 'Class 2 (virginica)']
    bins = np.linspace(measurements.min() - 0.15,
                       measurements.max() + 0.15, 28)
    for cls, color, name in zip([0, 1, 2], colors, class_names):
        ax3.hist(measurements[labels == cls], bins=bins,
                 color=color, alpha=0.75, label=name,
                 edgecolor='white', linewidth=0.4)
    ymax = ax3.get_ylim()[1]
    for tau, lbl in zip(best_thresholds, [r'$\tau_0$', r'$\tau_1$']):
        ax3.axvline(tau, color=RED, linestyle='--', linewidth=1.6, alpha=0.85)
        ax3.text(tau + 0.03, ymax * 0.93, lbl,
                 color=RED, fontsize=11, va='top')
    ax3.set_xlabel(r'Measurement')
    ax3.set_ylabel('Count')
    ax3.set_title(f'Linear Measurement Outcomes by Class\n'
                  f'Best run: {test_accs.max():.2%} test accuracy')
    ax3.legend(loc='upper left', framealpha=0.9)
    ax3.grid(False)  # Figure 3
    fig3.tight_layout()
    fig3.savefig('cv_pseudoinverse_measurement_dist.svg', bbox_inches='tight')
    fig3.savefig('cv_pseudoinverse_measurement_dist.png',
                 bbox_inches='tight', dpi=200)
    print("Saved cv_pseudoinverse_measurement_dist.svg/png")


# Main execution
if __name__ == "__main__":
    N_TRIALS       = 25
    EPOCHS         = 20
    ETA            = 0.1
    BATCH_SIZE     = 5
    CHECKPOINT_EVERY = 1   # acc_history is logged every epoch

    print("CV PSEUDOINVERSE LINEAR CLASSIFIER — MULTI-TRIAL")
    print("="*60)
    print(f"  Trials : {N_TRIALS}")
    print(f"  Epochs : {EPOCHS}  |  eta={ETA}  |  batch={BATCH_SIZE}")
    print(f"  Train  : {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    all_histories  = []
    test_accs      = []
    best_test_acc  = -1
    best_params    = None
    best_thresholds = None

    for trial_idx in range(N_TRIALS):
        seed = 42 + trial_idx * 13      # reproducible but varied seeds
        hist, tacc, params, thresholds = run_single_trial(
            seed, X_train, X_test, y_train, y_test,
            epochs=EPOCHS, eta=ETA, batch_size=BATCH_SIZE
        )
        all_histories.append(hist)
        test_accs.append(tacc)
        if tacc > best_test_acc:
            best_test_acc   = tacc
            best_params     = params.copy()
            best_thresholds = list(thresholds)

    test_accs = np.array(test_accs)
    print(f"\n{'='*60}")
    print(f"MULTI-TRIAL SUMMARY ({N_TRIALS} trials)")
    print(f"{'='*60}")
    print(f"  Test accuracy : {test_accs.mean():.1%} ± {test_accs.std():.1%}")
    print(f"  Best          : {test_accs.max():.1%}")
    print(f"  Worst         : {test_accs.min():.1%}")

    # Save best result
    import pickle
    save_data = {
        'params': best_params,
        'thresholds': best_thresholds,
        'test_accuracy': best_test_acc,
        'all_test_accuracies': test_accs,
        'all_histories': all_histories,
    }
    with open(f"cv_pseudoinverse_results_best{int(best_test_acc*100)}.pkl", "wb") as f:
        pickle.dump(save_data, f)
    print(f"Saved best model to pkl.")

    # Presentation figures
    plot_results(all_histories, test_accs, EPOCHS, CHECKPOINT_EVERY,
                 best_params, best_thresholds, X_test, y_test)