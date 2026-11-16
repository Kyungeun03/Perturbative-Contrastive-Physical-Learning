import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ===========================
# Parameters
# ===========================
np.random.seed(42)
learning_rate = 0.001
epochs = 500
target_energies = {0: 0.0193, 1: 0.0198, 2: 0.0203}  # Match actual energy distribution

# FEM Parameters
rows = 6
cols = 6
k_val = 1.0  # base stiffness

# ===========================
# Build Triangular Lattice Structure
# ===========================
def build_lattice():
    h = np.sqrt(3) / 2  # vertical spacing for equilateral triangles
    nodes = []
    for i in range(rows):
        y = i * h  # y increases upward
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
    
    # Boundary conditions
    fixed_nodes = [i for i, (x, y) in enumerate(nodes) if x == 0]  # fix left edge
    
    # Find nodes for force application (2 top + 2 right)
    max_y = max(node[1] for node in nodes)
    max_x = max(node[0] for node in nodes)
    
    top_nodes = [i for i, (x, y) in enumerate(nodes) if abs(y - max_y) < 1e-6]
    right_nodes = [i for i, (x, y) in enumerate(nodes) if abs(x - max_x) < 1e-6]
    
    # Select 2 top and 2 right nodes for force application
    force_nodes = []
    if len(top_nodes) >= 2:
        force_nodes.extend(sorted(top_nodes)[:2])  # Take first 2 top nodes
    if len(right_nodes) >= 2:
        # Take 2 right nodes that are not already in force_nodes
        available_right = [n for n in right_nodes if n not in force_nodes]
        force_nodes.extend(sorted(available_right)[:2])
    
    # Ensure we have exactly 4 force nodes
    force_nodes = force_nodes[:4]
    
    # Build edges
    edges = []
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            x1, y1 = nodes[i]
            x2, y2 = nodes[j]
            dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if np.isclose(dist, 1.0, atol=0.01):
                edges.append((i, j))
    
    return positions_init, n_nodes, fixed_nodes, force_nodes, edges

# ===========================
# FEM Solver
# ===========================
def solve_fem_baseline(positions_init, n_nodes, fixed_nodes, force_nodes, edges, iris_forces, mass_params):
    """Solve FEM with small perturbations from baseline"""
    ndof = 2 * n_nodes
    
    # Build stiffness matrix
    K = np.zeros((ndof, ndof))
    
    for (a, b) in edges:
        xi, yi = positions_init[a]
        xj, yj = positions_init[b]
        dx, dy = xj - xi, yj - yi
        L = np.sqrt(dx**2 + dy**2)
        c, s = dx / L, dy / L
        
        # Local stiffness matrix for truss element
        k_local = k_val * np.array([
            [ c*c,  c*s, -c*c, -c*s],
            [ c*s,  s*s, -c*s, -s*s],
            [-c*c, -c*s,  c*c,  c*s],
            [-c*s, -s*s,  c*s,  s*s]
        ])
        
        dofs = [2*a, 2*a+1, 2*b, 2*b+1]
        for i in range(4):
            for j in range(4):
                K[dofs[i], dofs[j]] += k_local[i, j]
    
    # Assemble force vector
    F = np.zeros(ndof)
    
    # Apply small-scale Iris forces (scaled appropriately)
    force_scale = 0.001
    for idx, node in enumerate(force_nodes):
        if idx < len(iris_forces):
            # Apply scaled forces in both directions
            F[2*node] += iris_forces[idx] * force_scale  # x-component
            F[2*node+1] += iris_forces[idx] * force_scale  # y-component
    
    # Add small mass-like effects
    for idx, node in enumerate(force_nodes):
        if idx < len(mass_params):
            F[2*node+1] -= mass_params[idx] * 0.001 * 9.8  # Small gravity-like effect
    
    # Apply boundary conditions
    fixed_dofs = []
    for node in fixed_nodes:
        fixed_dofs.extend([2*node, 2*node+1])
    
    # Additional constraint to remove rotation
    top_nodes = [i for i, (x, y) in enumerate(positions_init) 
                 if abs(y - np.max(positions_init[:, 1])) < 1e-6]
    if top_nodes:
        top_right_node = max(top_nodes, key=lambda i: positions_init[i][0])
        fixed_dofs.append(2 * top_right_node)  # Fix x-displacement
    
    free_dofs = np.setdiff1d(np.arange(ndof), fixed_dofs)
    
    try:
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        F_f = F[free_dofs]
        
        # Solve K * u = F
        u_f = np.linalg.solve(K_ff, F_f)
        u = np.zeros(ndof)
        u[free_dofs] = u_f
        
        # Compute deformed positions
        positions_equil = positions_init.copy()
        for i in range(n_nodes):
            positions_equil[i, 0] += u[2*i]
            positions_equil[i, 1] += u[2*i+1]
        
        # Calculate elastic energy
        elastic_energy = 0.0
        for (a, b) in edges:
            xi = positions_equil[a]
            xj = positions_equil[b]
            current_length = np.linalg.norm(xi - xj)
            rest_length = 1.0
            elastic_energy += 0.5 * k_val * (current_length - rest_length)**2
        
        # Calculate displacement magnitude
        displacements = np.sqrt(u[0::2]**2 + u[1::2]**2)
        max_disp = np.max(displacements)
        
        return elastic_energy, max_disp, u, positions_equil
        
    except np.linalg.LinAlgError:
        print("Warning: Singular matrix")
        return 0.0, 0.0, np.zeros(ndof), positions_init.copy()

# ===========================
# Load and preprocess Iris dataset
# ===========================
iris = load_iris()
X = iris.data
y = iris.target

# Normalize features to small scale
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Further scale to be in reasonable range for forces
X = X * 0.1  # Keep forces small

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

n_features = X_train.shape[1]  # 4 for Iris

# ===========================
# Initialize FEM structure and parameters
# ===========================
positions_init, n_nodes, fixed_nodes, force_nodes, edges = build_lattice()

print(f"Lattice created with {n_nodes} nodes and {len(edges)} edges")
print(f"Fixed nodes (left edge): {fixed_nodes}")
print(f"Force application nodes: {force_nodes}")

# Test baseline FEM solution
print(f"\nTesting baseline FEM solution...")
test_forces = np.array([0.01, 0.01, 0.01, 0.01])
test_masses = np.array([0.001, 0.001, 0.001, 0.001])
test_energy, test_disp, _, _ = solve_fem_baseline(
    positions_init, n_nodes, fixed_nodes, force_nodes, edges, test_forces, test_masses
)
print(f"Baseline test: Energy = {test_energy:.6f}, Max displacement = {test_disp:.6f}")

# Initialize learnable parameters
W = np.random.randn(n_features, n_features) * 0.001
W = (W + W.T) / 2  # Symmetrize
b = np.random.randn(n_features) * 0.001  # Small mass-like parameters
c = 0.0

# ===========================
# Training loop
# ===========================
print("\nStarting training...")
for epoch in range(epochs):
    total_loss = 0
    
    for i in range(len(X_train)):
        forces = X_train[i]  # Scaled Iris features as forces
        cls = y_train[i]
        target_energy = target_energies[cls]
        
        # Solve FEM with current parameters
        fem_energy, max_disp, displacements, positions_equil = solve_fem_baseline(
            positions_init, n_nodes, fixed_nodes, force_nodes, edges, forces, b
        )
        
        # Check for reasonable displacement (prevent explosion)
        if max_disp > 1.0:  # If displacement gets too large
            print(f"Warning: Large displacement {max_disp:.3f} at epoch {epoch}")
            break
        
        # Compute energy using the learned model
        E = forces @ W @ forces + np.dot(b, forces) + c
        
        # Combine with FEM energy (scaled appropriately)
        combined_energy = E + fem_energy
        
        # Loss
        loss = (combined_energy - target_energy) ** 2
        total_loss += loss
        
        # Gradients
        grad_E = 2 * (combined_energy - target_energy)
        dW = grad_E * np.outer(forces, forces)
        db = grad_E * forces
        dc = grad_E
        
        # Update parameters with small learning rate
        W -= learning_rate * dW
        b -= learning_rate * db
        c -= learning_rate * dc
        
        # Keep W symmetric and prevent explosion
        W = (W + W.T) / 2
        # Clip parameters to prevent instability
        W = np.clip(W, -1.0, 1.0)
        b = np.clip(b, -1.0, 1.0)
        c = np.clip(c, -10.0, 10.0)
    
    if epoch % 100 == 0:
        avg_loss = total_loss / len(X_train)
        print(f"Epoch {epoch}: Average Loss = {avg_loss:.6f}")

# ===========================
# Evaluation
# ===========================
print("\nEvaluating on test set...")
y_pred = []
test_energies = []
test_displacements = []

for i in range(len(X_test)):
    forces = X_test[i]
    
    # Solve FEM
    fem_energy, max_disp, _, _ = solve_fem_baseline(
        positions_init, n_nodes, fixed_nodes, force_nodes, edges, forces, b
    )
    
    test_displacements.append(max_disp)
    
    # Compute learned energy
    learned_energy = forces @ W @ forces + np.dot(b, forces) + c
    combined_energy = learned_energy + fem_energy
    
    test_energies.append(combined_energy)
    
    # Classify by finding closest target energy
    distances = [abs(combined_energy - target_energies[cls]) for cls in range(3)]
    predicted_class = np.argmin(distances)
    y_pred.append(predicted_class)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nFinal Test Accuracy: {accuracy:.4f}")
print(f"Test displacement range: {np.min(test_displacements):.6f} to {np.max(test_displacements):.6f}")

# ===========================
# Analysis and Visualization
# ===========================
print(f"\nLearned Parameters:")
print(f"W matrix (stiffness-like):\n{W}")
print(f"b vector (mass-like): {b}")
print(f"c parameter: {c}")

# Show example FEM solution
print(f"\nExample FEM solution with first test sample:")
example_forces = X_test[0]
example_energy, example_max_disp, example_displacements, example_positions = solve_fem_baseline(
    positions_init, n_nodes, fixed_nodes, force_nodes, edges, example_forces, b
)

print(f"Applied forces: {example_forces}")
print(f"FEM elastic energy: {example_energy:.6f}")
print(f"Max displacement: {example_max_disp:.6f}")

# Plot results
plt.figure(figsize=(15, 5))

# Energy distribution
plt.subplot(1, 3, 1)
for cls in range(3):
    class_energies = [test_energies[i] for i in range(len(y_test)) if y_test[i] == cls]
    plt.hist(class_energies, alpha=0.7, label=f'Class {cls} (target: {target_energies[cls]})', bins=10)
plt.xlabel('Combined Energy')
plt.ylabel('Frequency')
plt.title('Energy Distribution by Class')
plt.legend()
plt.grid(True, alpha=0.3)

# Displacement vs Energy
plt.subplot(1, 3, 2)
colors = ['red', 'green', 'blue']
for cls in range(3):
    class_indices = [i for i in range(len(y_test)) if y_test[i] == cls]
    class_disps = [test_displacements[i] for i in class_indices]
    class_energies = [test_energies[i] for i in class_indices]
    plt.scatter(class_disps, class_energies, c=colors[cls], alpha=0.7, label=f'Class {cls}')
plt.xlabel('Max Displacement')
plt.ylabel('Combined Energy')
plt.title('Displacement vs Energy')
plt.legend()
plt.grid(True, alpha=0.3)

# Example deformed lattice (properly scaled)
plt.subplot(1, 3, 3)
# Original lattice
for (a, b) in edges:
    plt.plot([positions_init[a, 0], positions_init[b, 0]],
             [positions_init[a, 1], positions_init[b, 1]], 
             'b-', linewidth=0.8, alpha=0.5)

# Deformed lattice
for (a, b) in edges:
    plt.plot([example_positions[a, 0], example_positions[b, 0]],
             [example_positions[a, 1], example_positions[b, 1]], 
             'r-', linewidth=1.2)

plt.scatter(positions_init[:, 0], positions_init[:, 1], 
           color='blue', s=20, alpha=0.5, label='Initial')
plt.scatter(example_positions[:, 0], example_positions[:, 1], 
           color='red', s=30, label='Deformed')

# Highlight force application nodes
for idx, node in enumerate(force_nodes):
    plt.scatter(positions_init[node, 0], positions_init[node, 1], 
               color='green', s=80, marker='*', 
               label='Force Nodes' if idx == 0 else "")

plt.title(f'Example: Class {y_test[0]} (Max disp: {example_max_disp:.4f})')
plt.axis('equal')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ===========================
# Detailed Results
# ===========================
print(f"\n=== Classification Results ===")
for cls in range(3):
    class_indices = [i for i in range(len(y_test)) if y_test[i] == cls]
    class_predictions = [y_pred[i] for i in class_indices]
    class_accuracy = sum(1 for pred in class_predictions if pred == cls) / len(class_predictions) if class_predictions else 0
    print(f"Class {cls}: {len(class_indices)} samples, accuracy: {class_accuracy:.3f}")

print(f"\nScale verification:")
print(f"- Force scale: {np.max(np.abs(X_test)):.4f}")
print(f"- Displacement scale: {np.max(test_displacements):.4f}")
print(f"- Energy scale: {np.max(test_energies):.4f}")