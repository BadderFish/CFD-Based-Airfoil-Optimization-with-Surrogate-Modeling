import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

# ── REBUILD THE SURROGATE (same as surrogate_model.py) ────
# We retrain on ALL 13 points this time — no holdout needed
# The validation was already done in surrogate_model.py
# Now we want the best possible surrogate for optimization

df = pd.read_csv("sweep_results.csv")

X = df[["alpha", "Re"]].values
y_Cl = df["Cl"].values
y_Cd = df["Cd"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)

gp_Cl = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
gp_Cd = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)

gp_Cl.fit(X_scaled, y_Cl)
gp_Cd.fit(X_scaled, y_Cd)

print("Surrogate rebuilt on all 13 points.\n")


# ── DEFINE THE OBJECTIVE FUNCTION ─────────────────────────
# scipy.optimize MINIMIZES by default
# We want to MAXIMIZE L/D = Cl/Cd
# So we minimize the NEGATIVE of L/D
# This is a standard trick — minimizing -f(x) = maximizing f(x)

def negative_LD(params):
    alpha, re = params
    X_input = scaler.transform([[alpha, re]])
    Cl = gp_Cl.predict(X_input)[0]
    Cd = gp_Cd.predict(X_input)[0]

    # Guard against Cd going negative (GP can predict nonsense outside data range)
    if Cd <= 0 or Cl <= 0:
        return 0.0  # return bad value to steer optimizer away

    return -(Cl / Cd)  # negative because we're minimizing


# ── SET BOUNDS ────────────────────────────────────────────
# Keep the optimizer inside our data range
# Extrapolating outside this is unreliable
bounds = [
    (-2.0, 12.0),          # AoA bounds (degrees)
    (500000, 2000000)      # Reynolds number bounds
]

# ── RUN THE OPTIMIZER ─────────────────────────────────────
# We run from multiple starting points to avoid local minima
# A GP surface can have bumps — one starting point isn't enough
print("Running optimizer from multiple starting points...")

best_result = None
best_LD = -np.inf

starting_points = [
    [4.0,  500000],
    [4.0,  1000000],
    [4.0,  2000000],
    [6.0,  1000000],
    [6.0,  2000000],
    [8.0,  2000000],
]

for x0 in starting_points:
    result = minimize(
        negative_LD,
        x0=x0,
        bounds=bounds,
        method='L-BFGS-B'   # gradient-based method, works well for smooth surfaces
    )
    ld = -result.fun  # convert back from negative
    if ld > best_LD:
        best_LD = ld
        best_result = result

optimal_alpha = best_result.x[0]
optimal_re    = best_result.x[1]

print(f"\n{'='*45}")
print("OPTIMIZATION RESULT")
print(f"{'='*45}")
print(f"Optimal AoA      : {optimal_alpha:.3f} degrees")
print(f"Optimal Re       : {optimal_re:,.0f}")
print(f"Predicted L/D    : {best_LD:.3f}")

# Get individual Cl and Cd at optimal point
X_opt = scaler.transform([[optimal_alpha, optimal_re]])
Cl_opt = gp_Cl.predict(X_opt)[0]
Cd_opt = gp_Cd.predict(X_opt)[0]
print(f"Predicted Cl     : {Cl_opt:.4f}")
print(f"Predicted Cd     : {Cd_opt:.5f}")
print(f"{'='*45}\n")


# ── PLOT: L/D SURFACE WITH OPTIMAL POINT MARKED ───────────
alpha_grid = np.linspace(-2, 12, 100)
re_grid    = np.linspace(500000, 2000000, 100)
AA, RR     = np.meshgrid(alpha_grid, re_grid)

X_grid        = np.column_stack([AA.ravel(), RR.ravel()])
X_grid_scaled = scaler.transform(X_grid)

Cl_surf = gp_Cl.predict(X_grid_scaled).reshape(AA.shape)
Cd_surf = gp_Cd.predict(X_grid_scaled).reshape(AA.shape)
LD_surf = Cl_surf / np.maximum(Cd_surf, 1e-6)  # avoid division by zero

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: 2D contour map of L/D surface
ax = axes[0]
contour = ax.contourf(AA, RR/1e6, LD_surf, levels=30, cmap='RdYlGn')
plt.colorbar(contour, ax=ax, label='L/D')
ax.scatter(df["alpha"], df["Re"]/1e6,
           color='blue', s=60, zorder=5, label='XFOIL data points')
ax.scatter(optimal_alpha, optimal_re/1e6,
           color='red', s=200, marker='*', zorder=6, label=f'Optimal: AoA={optimal_alpha:.1f}°, Re={optimal_re/1e6:.2f}M')
ax.set_xlabel('AoA (degrees)')
ax.set_ylabel('Reynolds Number (×10⁶)')
ax.set_title('L/D Surrogate Surface\nwith Optimal Point')
ax.legend(fontsize=9)

# Right: Cl/Cd curves at optimal Re
alphas = np.linspace(-2, 12, 100)
X_line = scaler.transform([[a, optimal_re] for a in alphas])
Cl_line = gp_Cl.predict(X_line)
Cd_line = gp_Cd.predict(X_line)
LD_line = Cl_line / np.maximum(Cd_line, 1e-6)

ax2 = axes[1]
ax2.plot(alphas, LD_line, 'b-', linewidth=2, label=f'Surrogate at Re={optimal_re/1e6:.1f}M')
ax2.axvline(optimal_alpha, color='red', linestyle='--', label=f'Optimal AoA = {optimal_alpha:.1f}°')
ax2.scatter(df[df["Re"]==int(round(optimal_re, -5))]["alpha"],
            df[df["Re"]==int(round(optimal_re, -5))]["LD"],
            color='black', s=60, zorder=5, label='XFOIL data points')
ax2.set_xlabel('AoA (degrees)')
ax2.set_ylabel('L/D')
ax2.set_title(f'L/D vs AoA at Optimal Re\nMax L/D = {best_LD:.1f} at AoA = {optimal_alpha:.1f}°')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("optimizer_result.png", dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved as optimizer_result.png")