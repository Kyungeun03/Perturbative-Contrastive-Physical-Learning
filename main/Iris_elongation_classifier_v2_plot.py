import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------
# Parameters
# ---------------------------
np.random.seed(42)
epochs = 200
learning_rate = 0.005
target_energies = {0: 0.0193, 1: 0.0198, 2: 0.0203}

rows, cols = 6, 6
k_val = 1.0

# ---------------------------
# Build Triangular Lattice
# ---------------------------
def build_lattice():
    h = np.sqrt(3) / 2
    nodes = []
    for i in range(rows):
        y = i * h
        if i % 2 == 0:
            for j in range(cols):
                x = j
                nodes.append((x, y))
        else:
            for j in range(cols - 1):
                x = j + 0.5
                nodes.append((x, y))
    positions_init = np.array(nodes)
    n_nodes = len(nodes)
    fixed_nodes = [i for i, (x, y) in enumerate(nodes) if x == 0]

    max_y = max(node[1] for node in nodes)
    max_x = max(node[0] for node in nodes)
    top_nodes = [i for i, (x, y) in enumerate(nodes) if abs(y - max_y) < 1e-6]
    right_nodes = [i for i, (x, y) in enumerate(nodes) if abs(x - max_x) < 1e-6]
    force_nodes = (top_nodes[:2] + [n for n in right_nodes if n not in top_nodes][:2])[:4]

    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            x1, y1 = nodes[i]
            x2, y2 = nodes[j]
            if np.isclose(np.sqrt((x1 - x2)**2 + (y1 - y2)**2), 1.0, atol=0.01):
                edges.append((i, j))
    return positions_init, n_nodes, fixed_nodes, force_nodes, edges

# ---------------------------
# FEM Solver
# ---------------------------
def solve_fem(positions_init, n_nodes, fixed_nodes, force_nodes, edges, iris_forces, mass_params):
    ndof = 2 * n_nodes
    K = np.zeros((ndof, ndof))

    for (a, b) in edges:
        xi, yi = positions_init[a]
        xj, yj = positions_init[b]
        dx, dy = xj - xi, yj - yi
        L = np.sqrt(dx**2 + dy**2)
        c, s = dx / L, dy / L
        k_local = k_val * np.array([
            [c*c, c*s, -c*c, -c*s],
            [c*s, s*s, -c*s, -s*s],
            [-c*c, -c*s, c*c, c*s],
            [-c*s, -s*s, c*s, s*s]
        ])
        dofs = [2*a, 2*a+1, 2*b, 2*b+1]
        for i1 in range(4):
            for j1 in range(4):
                K[dofs[i1], dofs[j1]] += k_local[i1, j1]

    F = np.zeros(ndof)
    force_scale = 0.001
    for idx, node in enumerate(force_nodes):
        if idx < len(iris_forces):
            F[2*node] += iris_forces[idx] * force_scale
            F[2*node+1] += iris_forces[idx] * force_scale
    for idx, node in enumerate(force_nodes):
        if idx < len(mass_params):
            F[2*node+1] -= mass_params[idx] * 0.001 * 9.8

    fixed_dofs = []
    for node in fixed_nodes:
        fixed_dofs.extend([2*node, 2*node+1])
    top_nodes = [i for i, (x, y) in enumerate(positions_init) if abs(y - np.max(positions_init[:, 1])) < 1e-6]
    if top_nodes:
        top_right_node = max(top_nodes, key=lambda i: positions_init[i][0])
        fixed_dofs.append(2 * top_right_node)

    free_dofs = np.setdiff1d(np.arange(ndof), fixed_dofs)
    try:
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        F_f = F[free_dofs]
        u_f = np.linalg.solve(K_ff, F_f)
        u = np.zeros(ndof)
        u[free_dofs] = u_f
        positions_equil = positions_init.copy()
        for i in range(n_nodes):
            positions_equil[i, 0] += u[2*i]
            positions_equil[i, 1] += u[2*i+1]
        elastic_energy = sum(0.5 * k_val * (np.linalg.norm(positions_equil[a]-positions_equil[b]) - 1.0)**2 for (a, b) in edges)
        displacements = np.sqrt(u[0::2]**2 + u[1::2]**2)
        max_disp = np.max(displacements)
        return elastic_energy, max_disp
    except np.linalg.LinAlgError:
        return 0.0, 0.0

# ---------------------------
# Load Iris Dataset
# ---------------------------
iris = load_iris()
X = iris.data
y = iris.target
scaler = StandardScaler()
X = scaler.fit_transform(X) * 0.1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

positions_init, n_nodes, fixed_nodes, force_nodes, edges = build_lattice()

# ---------------------------
# Initialize Model Parameters
# ---------------------------
W = np.random.randn(X_train.shape[1], X_train.shape[1]) * 0.001
W = (W + W.T) / 2
b = np.random.randn(X_train.shape[1]) * 0.001
c = 0.0

# ---------------------------
# Training Loop
# ---------------------------
from numpy.linalg import pinv

for epoch in range(epochs):
    J_rows = []
    delta_z = []
    
    for i in range(len(X_train)):
        forces = X_train[i]
        cls = y_train[i]
        target_energy = target_energies[cls]

        # Compute combined energy
        fem_energy, _ = solve_fem(positions_init, n_nodes, fixed_nodes, force_nodes, edges, forces, b)
        E = forces @ W @ forces + np.dot(b, forces) + c
        combined_energy = E + fem_energy

        # Error
        error = target_energy - combined_energy
        delta_z.append(error)

        # Build Jacobian row: vec(forces outer product), forces, 1
        outer_ff = np.outer(forces, forces).flatten()
        J_row = np.concatenate([outer_ff, forces, [1]])
        J_rows.append(J_row)

    J = np.array(J_rows)
    delta_z = np.array(delta_z)

    # Compute pseudoinverse update
    J_pinv = pinv(J)
    delta_theta = J_pinv @ delta_z

    # Extract components
    size_W = W.shape[0] * W.shape[1]
    delta_W = delta_theta[:size_W].reshape(W.shape)
    delta_b = delta_theta[size_W:size_W + len(b)]
    delta_c = delta_theta[-1]

    # Update parameters
    W += learning_rate * delta_W
    b += learning_rate * delta_b
    c += learning_rate * delta_c

    # Symmetrize W
    W = (W + W.T) / 2

    if epoch % 10 == 0:
        avg_error = np.mean(np.abs(delta_z))
        print(f"Epoch {epoch}: Avg |Error| = {avg_error:.6f}")

# ---------------------------
# Test & Collect Data
# ---------------------------
test_energies, test_displacements, y_pred = [], [], []
for i in range(len(X_test)):
    forces = X_test[i]
    fem_energy, max_disp = solve_fem(positions_init, n_nodes, fixed_nodes, force_nodes, edges, forces, b)
    learned_energy = forces @ W @ forces + np.dot(b, forces) + c
    combined_energy = learned_energy + fem_energy
    test_energies.append(combined_energy)
    test_displacements.append(max_disp)
    distances = [abs(combined_energy - target_energies[cls]) for cls in range(3)]
    y_pred.append(np.argmin(distances))

print("\nTest Accuracy (energy-based):", accuracy_score(y_test, y_pred))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# ---- Compute train cluster energy & displacement ----
train_energies = []
train_displacements = []
for i in range(len(X_train)):
    forces = X_train[i]
    fem_energy, max_disp = solve_fem(positions_init, n_nodes, fixed_nodes, force_nodes, edges, forces, b)
    learned_energy = forces @ W @ forces + np.dot(b, forces) + c
    combined_energy = learned_energy + fem_energy
    train_energies.append(combined_energy)
    train_displacements.append(max_disp)

# Group by class
E0 = [train_energies[i] for i in range(len(y_train)) if y_train[i] == 0]
E1 = [train_energies[i] for i in range(len(y_train)) if y_train[i] == 1]
E2 = [train_energies[i] for i in range(len(y_train)) if y_train[i] == 2]

D1 = [train_displacements[i] for i in range(len(y_train)) if y_train[i] == 1]
D2 = [train_displacements[i] for i in range(len(y_train)) if y_train[i] == 2]

# ---- Compute proper thresholds ----
#energy_boundary = (max(E0) + np.mean(E1 + E2)) / 2
energy_boundary = (min(E0) + max(max(E1), max(E2))) / 2
disp_boundary = (max(D1) + min(D2)) / 2
class0_is_low = np.mean(E0) < np.mean(E2)

print(f"Energy boundary = {energy_boundary:.6f}")
print(f"Displacement boundary = {disp_boundary:.6f}")

# ---- Apply decision rule on TEST data ----
y_pred = []
for e_val, d_val in zip(test_energies, test_displacements):
    if d_val < disp_boundary:
        y_pred.append(1)
    else:
        y_pred.append(0 if (e_val < energy_boundary) == class0_is_low else 2)

linear_acc = accuracy_score(y_test, y_pred)
print(f"Linear Boundary Accuracy: {linear_acc:.4f}")
# ---------------------------
# Visualization
# ---------------------------
plt.figure(figsize=(6, 5))
colors = ['red', 'green', 'blue']
for cls in range(3):
    indices = [i for i in range(len(y_test)) if y_test[i] == cls]
    plt.scatter([test_displacements[i] for i in indices], [test_energies[i] for i in indices], c=colors[cls], label=f'Class {cls}')
plt.axhline(y=energy_boundary, color='black', linestyle='--', label='Boundary of the function F')
plt.axvline(x=disp_boundary, color='gray', linestyle='--', label='Displacement boundary')
plt.xlabel('Max Displacement')
plt.ylabel('F values')
plt.title('Displacement vs F values with Decision Boundaries')
plt.legend()
plt.grid(True)
plt.show()

# ---- Compute accuracy ----
accuracy = accuracy_score(y_test, y_pred)
print(f"\nPhysical Rule Test Accuracy: {accuracy:.4f}")

# ---- Confusion matrix ----
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0","Class 1","Class 2"], yticklabels=["Class 0","Class 1","Class 2"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Physical Classifier)")
plt.show()

# ---- Plot with correct boundaries ----
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']
for cls in range(3):
    idxs = [i for i in range(len(y_test)) if y_test[i] == cls]
    plt.scatter([test_displacements[i] for i in idxs],
                [test_energies[i] for i in idxs],
                color=colors[cls], label=f'Class {cls}', alpha=0.7)

# Add boundaries
plt.axhline(y=energy_boundary, color='black', linestyle='--', label='Energy boundary')
plt.axvline(x=disp_boundary, color='gray', linestyle='--', label='Displacement boundary')

plt.xlabel('Max Displacement')
plt.ylabel('Combined Energy')
plt.title('Displacement vs Energy with Decision Boundaries')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ---- Plot with correct boundaries ----
import matplotlib.patches as mpatches
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9, 'axes.labelsize': 10,
    'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})
PUB_COLORS  = {0: '#E63946', 1: "#000000", 2: "#0800FF"}
PUB_MARKERS = {0: 'o', 1: 's', 2: '^'}
PUB_LABELS  = {0: 'Iris setosa', 1: 'Iris versicolor', 2: 'Iris virginica'}

fig, ax = plt.subplots(figsize=(3.5, 3.2))
for cls in range(3):
    idxs = [i for i in range(len(y_test)) if y_test[i] == cls]
    ax.scatter([test_displacements[i] for i in idxs],
               [test_energies[i] for i in idxs],
               c=PUB_COLORS[cls], marker=PUB_MARKERS[cls],
               s=22, linewidths=0.3, edgecolors='white', zorder=3,
               label=PUB_LABELS[cls])
ax.axhline(y=energy_boundary, color='#222222', lw=0.9, ls='--', zorder=2, label=r'$\mathcal{F}$ boundary')
ax.axvline(x=disp_boundary,   color='#888888', lw=0.9, ls=':',  zorder=2, label=r'$u_\mathrm{max}$ boundary')
ax.set_xlabel(r'Max displacement, $u_\mathrm{max}$')
ax.set_ylabel(r'Combined energy, $\mathcal{F}$')
ax.legend(frameon=False, fontsize=7.5)
fig.tight_layout()
ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.savefig('classification2.svg', bbox_inches='tight')
plt.show()
