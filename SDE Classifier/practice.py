import numpy as np
import matplotlib.pyplot as plt

# Parameters
x0 = 0.1          # Initial condition
T = 5.0           # Total time
dt = 0.01         # Timestep
sigma = 0.1       # Noise intensity
num_realizations = 5  # Number of stochastic realizations

# Discretization
N = int(T / dt)
t = np.linspace(0, T, N+1)

# Euler-Maruyama method
np.random.seed(42)  # For reproducibility
X = np.zeros((num_realizations, N+1))
X[:, 0] = x0

for i in range(num_realizations):
    for n in range(N):
        dW = np.random.normal(0, np.sqrt(dt))  # Wiener increment
        X[i, n+1] = X[i, n] + X[i, n] * (1 - X[i, n]) * dt + sigma * dW

# Plot realizations
plt.figure(figsize=(10, 6))
for i in range(num_realizations):
    plt.plot(t, X[i, :], lw=1, label=f'Realization {i+1}')

# Plot deterministic solution (for comparison)
x_det = np.exp(t) / (9 + np.exp(t))
plt.plot(t, x_det, 'k--', lw=2, label='Deterministic Solution')

plt.xlabel('Time (t)')
plt.ylabel('X(t)')
plt.title(f'Logistic SDE: {num_realizations} Realizations (σ={sigma}, Δt={dt})')
plt.legend()
plt.grid(True)
plt.show()


# Parameters
X0 = 0.0          # Initial condition
T = 1.0           # Total time
dt = 0.001        # Timestep
N = int(T / dt)   # Number of steps
t = np.linspace(0, T, N+1)  # Time grid

# Euler-Maruyama for dX_t = dW_t
np.random.seed(42)  # Reproducibility
dW = np.random.normal(0, np.sqrt(dt), size=N)  # Wiener increments
X = np.zeros(N+1)
X[0] = X0

for n in range(N):
    X[n+1] = X[n] + dW[n]  # Since f=0, g=1

# Plot
plt.figure(figsize=(10, 6))
plt.plot(t, X, lw=1, label=f'Numerical Solution (Δt={dt})')
plt.xlabel('Time (t)')
plt.ylabel('X(t)')
plt.title('Numerical Solution of dX_t = dW_t (Wiener Process)')
plt.grid(True)
plt.legend()
plt.show()

num_realizations = 100
X_multi = np.zeros((num_realizations, N+1))

for i in range(num_realizations):
    dW = np.random.normal(0, np.sqrt(dt), size=N)
    X_multi[i, 0] = X0
    for n in range(N):
        X_multi[i, n+1] = X_multi[i, n] + dW[n]

# Plot
plt.figure(figsize=(10, 6))
for i in range(num_realizations):
    plt.plot(t, X_multi[i], lw=1)
plt.xlabel('Time (t)')
plt.ylabel('X(t)')
plt.title(f'{num_realizations} Realizations of dX_t = dW_t')
plt.grid(True)
plt.legend()
plt.show()

# Parameters
X0 = 0.0          # Initial condition
T = 10.0           # Total time
dt = 0.001        # Timestep
N = int(T / dt)   # Number of steps
t = np.linspace(0, T, N+1)  # Time grid
num_realizations = 5  # Number of paths
alpha = 0.5       # Opacity for visualization

# Generate and plot realizations
plt.figure(figsize=(10, 6))
for i in range(num_realizations):
    dW = np.random.normal(0, np.sqrt(dt), size=N)  # Wiener increments
    X = X0 + np.cumsum(np.insert(1*dt + 1*dW, 0, X0))  # Euler-Maruyama integration
    plt.plot(t, X, lw=1, alpha=alpha, label=f'Path {i+1}')

# Theoretical mean (X0 + t)
plt.plot(t, X0 + t, 'k--', lw=2, label='Theoretical Mean')

plt.title(f'{num_realizations} Realizations of $dX_t = dt + dW_t$')
plt.xlabel('Time (t)')
plt.ylabel('X(t)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Parameters
X0 = 0.0          # Initial condition
T = 10.0           # Total time
dt = 0.001        # Timestep
N = int(T / dt)   # Number of steps
t = np.linspace(0, T, N+1)  # Time grid
num_mc = 1000     # Number of Monte Carlo realizations

# Storage for all paths
all_paths = np.zeros((num_mc, N+1))
all_paths[:, 0] = X0

# Monte Carlo simulation
for i in range(num_mc):
    dW = np.random.normal(0, np.sqrt(dt), size=N)  # Wiener increments
    for n in range(N):
        all_paths[i, n+1] = all_paths[i, n] + 1*dt + 1*dW[n]  # Euler-Maruyama

# Compute statistics
mean_X = np.mean(all_paths, axis=0)
variance_X = np.var(all_paths, axis=0)

# Theoretical values
theoretical_mean = X0 + t
theoretical_variance = t  # Since Var(W_t) = t

# Plotting
plt.figure(figsize=(12, 6))

# Plot mean ± std
plt.subplot(1, 2, 1)
plt.plot(t, mean_X, 'b-', lw=2, label='Empirical Mean')
plt.plot(t, theoretical_mean, 'k--', lw=2, label='Theoretical Mean')
plt.fill_between(t, 
                 mean_X - np.sqrt(variance_X),
                 mean_X + np.sqrt(variance_X),
                 color='blue', alpha=0.2, label='±1 Std Dev')
plt.xlabel('Time (t)')
plt.ylabel('Mean')
plt.title('Mean ± Standard Deviation')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot variance
plt.subplot(1, 2, 2)
plt.plot(t, variance_X, 'r-', lw=2, label='Empirical Variance')
plt.plot(t, theoretical_variance, 'k--', lw=2, label='Theoretical Variance')
plt.xlabel('Time (t)')
plt.ylabel('Variance')
plt.title('Variance over Time')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()