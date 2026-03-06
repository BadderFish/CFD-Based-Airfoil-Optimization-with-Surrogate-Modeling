import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ── LOAD DATA ─────────────────────────────────────────────
df = pd.read_csv("sweep_results.csv")
print(f"Loaded {len(df)} data points from sweep_results.csv\n")

# ── FEATURES AND TARGETS ──────────────────────────────────
# Inputs to the model: AoA and Reynolds number
# Outputs: Cl and Cd (we train two separate GP models)
X = df[["alpha", "Re"]].values
y_Cl = df["Cl"].values
y_Cd = df["Cd"].values

# ── HOLDOUT 3 CASES FOR VALIDATION ───────────────────────
# We fix the holdout indices so results are reproducible
# Choosing indices spread across the dataset (low/mid/high AoA)
holdout_idx = [2, 6, 10]
train_idx = [i for i in range(len(df)) if i not in holdout_idx]

X_train = X[train_idx]
X_test  = X[holdout_idx]

y_Cl_train = y_Cl[train_idx]
y_Cl_test  = y_Cl[holdout_idx]

y_Cd_train = y_Cd[train_idx]
y_Cd_test  = y_Cd[holdout_idx]

print("Holdout cases (withheld from training):")
print(df.iloc[holdout_idx][["alpha", "Re", "Cl", "Cd"]].to_string(index=False))
print()

# ── SCALE INPUTS ──────────────────────────────────────────
# AoA ranges from -2 to 12, Re ranges from 500000 to 2000000
# Without scaling, Re dominates and the GP learns nothing useful
# StandardScaler makes both features zero-mean, unit-variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── DEFINE AND TRAIN GP MODELS ────────────────────────────
# Matern kernel is standard for engineering surrogate models
# It's more flexible than pure RBF for non-smooth physical responses
kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)

gp_Cl = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=10,  # tries 10 different starting points to find best fit
    normalize_y=True
)

gp_Cd = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=10,
    normalize_y=True
)

print("Training GP model for Cl...")
gp_Cl.fit(X_train_scaled, y_Cl_train)

print("Training GP model for Cd...")
gp_Cd.fit(X_train_scaled, y_Cd_train)
print()

# ── VALIDATE ON HOLDOUT CASES ─────────────────────────────
Cl_pred, Cl_std = gp_Cl.predict(X_test_scaled, return_std=True)
Cd_pred, Cd_std = gp_Cd.predict(X_test_scaled, return_std=True)

print("=" * 55)
print("VALIDATION RESULTS (holdout cases)")
print("=" * 55)
print("\nCl predictions:")
print(f"{'AoA':>6} {'Re':>10} {'Actual':>10} {'Predicted':>10} {'Error':>10}")
for i in range(len(holdout_idx)):
    alpha_val = X_test[i][0]
    re_val    = int(X_test[i][1])
    actual    = y_Cl_test[i]
    predicted = Cl_pred[i]
    error     = abs(actual - predicted)
    print(f"{alpha_val:>6.1f} {re_val:>10,} {actual:>10.4f} {predicted:>10.4f} {error:>10.4f}")

rmse_Cl = np.sqrt(mean_squared_error(y_Cl_test, Cl_pred))
print(f"\nCl RMSE on holdout: {rmse_Cl:.4f}")

print("\nCd predictions:")
print(f"{'AoA':>6} {'Re':>10} {'Actual':>10} {'Predicted':>10} {'Error':>10}")
for i in range(len(holdout_idx)):
    alpha_val = X_test[i][0]
    re_val    = int(X_test[i][1])
    actual    = y_Cd_test[i]
    predicted = Cd_pred[i]
    error     = abs(actual - predicted)
    print(f"{alpha_val:>6.1f} {re_val:>10,} {actual:>10.4f} {predicted:>10.4f} {error:>10.4f}")

rmse_Cd = np.sqrt(mean_squared_error(y_Cd_test, Cd_pred))
print(f"\nCd RMSE on holdout: {rmse_Cd:.4f}")
print("=" * 55)

# ── GENERATE SURROGATE SURFACE FOR PLOTTING ───────────────
# Create a fine grid of AoA and Re values
alpha_grid = np.linspace(-2, 12, 50)
re_grid    = np.linspace(500000, 2000000, 50)
AA, RR     = np.meshgrid(alpha_grid, re_grid)

# Flatten, scale, predict, reshape
X_grid        = np.column_stack([AA.ravel(), RR.ravel()])
X_grid_scaled = scaler.transform(X_grid)

Cl_surface = gp_Cl.predict(X_grid_scaled).reshape(AA.shape)
Cd_surface = gp_Cd.predict(X_grid_scaled).reshape(AA.shape)
LD_surface = Cl_surface / Cd_surface

# ── PLOT 1: Cl SURROGATE SURFACE ──────────────────────────
fig = plt.figure(figsize=(16, 5))

ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(AA, RR/1e6, Cl_surface, cmap='viridis', alpha=0.8)
ax1.scatter(X_train[:,0], X_train[:,1]/1e6, y_Cl_train,
            color='red', s=50, zorder=5, label='Training data')
ax1.scatter(X_test[:,0], X_test[:,1]/1e6, y_Cl_test,
            color='orange', s=80, marker='^', zorder=5, label='Holdout data')
ax1.set_xlabel('AoA (deg)')
ax1.set_ylabel('Re (×10⁶)')
ax1.set_zlabel('Cl')
ax1.set_title(f'Cl Surrogate Surface\nRMSE = {rmse_Cl:.4f}')
ax1.legend(fontsize=8)

# ── PLOT 2: Cd SURROGATE SURFACE ──────────────────────────
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(AA, RR/1e6, Cd_surface, cmap='plasma', alpha=0.8)
ax2.scatter(X_train[:,0], X_train[:,1]/1e6, y_Cd_train,
            color='red', s=50, zorder=5, label='Training data')
ax2.scatter(X_test[:,0], X_test[:,1]/1e6, y_Cd_test,
            color='orange', s=80, marker='^', zorder=5, label='Holdout data')
ax2.set_xlabel('AoA (deg)')
ax2.set_ylabel('Re (×10⁶)')
ax2.set_zlabel('Cd')
ax2.set_title(f'Cd Surrogate Surface\nRMSE = {rmse_Cd:.4f}')
ax2.legend(fontsize=8)

# ── PLOT 3: L/D SURFACE ───────────────────────────────────
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(AA, RR/1e6, LD_surface, cmap='coolwarm', alpha=0.8)
ax3.set_xlabel('AoA (deg)')
ax3.set_ylabel('Re (×10⁶)')
ax3.set_zlabel('L/D')
ax3.set_title('L/D Surrogate Surface\n(Cl/Cd predicted)')

plt.tight_layout()
plt.savefig("surrogate_surface.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as surrogate_surface.png")
