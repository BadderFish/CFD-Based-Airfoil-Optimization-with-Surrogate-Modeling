# CFD-Based Airfoil Optimization with Surrogate Modeling

## Problem Statement

Traditional CFD-based airfoil optimization is computationally expensive, requiring hundreds of flow simulations to explore the design space and identify optimal operating conditions. For aircraft design and performance analysis, engineers need to rapidly determine the maximum lift-to-drag ratio across varying angles of attack and Reynolds numbers without running exhaustive CFD campaigns.

## Solution

This project implements a machine learning-accelerated optimization pipeline that reduces the computational cost of airfoil performance optimization by 95%. Instead of running hundreds of XFOIL simulations, we:

1. Generate a strategic dataset of 13 CFD simulations covering the design space
2. Train a Gaussian Process surrogate model to predict aerodynamic coefficients (Cl, Cd) at any operating condition
3. Use gradient-based optimization on the surrogate to find maximum L/D in seconds rather than hours

## Technical Approach

**Phase 1: CFD Data Generation**
- Automated XFOIL 6.99 simulations for NACA 2412 airfoil
- Parameter space: 5 angles of attack (-2°, 0°, 4°, 8°, 12°) × 3 Reynolds numbers (500k, 1M, 2M)
- 13 of 15 runs converged (2 failures at edge cases due to XFOIL boundary layer limitations)
- Output: lift coefficient (Cl), drag coefficient (Cd), moment coefficient (Cm)

**Phase 2: Surrogate Model Training**
- Gaussian Process Regressor with Matérn kernel (ν=2.5) for smooth aerodynamic surfaces
- 10 training points, 3 held out for validation
- Validation performance: Cl RMSE = 0.0184, Cd RMSE = 0.0013
- Model captures complex nonlinear relationships between flow conditions and aerodynamic forces

**Phase 3: Optimization**
- Objective: maximize lift-to-drag ratio (L/D)
- Method: scipy.optimize L-BFGS-B with 6 multi-start initializations to avoid local minima
- Constraint: Reynolds number ≤ 2,000,000 (hardware limitation boundary)

## Results

**Optimal Operating Condition:**
- Angle of Attack: 4.654°
- Reynolds Number: 2,000,000
- Maximum L/D: 115.754
- Corresponding Cl: 0.7608
- Corresponding Cd: 0.00657

**Physical Validation:**
The result is consistent with known NACA 2412 aerodynamic behavior. Peak efficiency occurs at moderate angle of attack where lift generation is substantial but the airfoil has not yet entered the high-drag pre-stall regime. Operating at maximum Reynolds number (within constraints) minimizes viscous drag effects.

## Requirements

**Software:**
- XFOIL 6.99
- Python 3.14
- pandas
- scikit-learn
- scipy
- matplotlib
- numpy

**Hardware:**
- Any modern CPU (XFOIL is single-threaded)
- ~10 MB disk space for data outputs

## Usage

**Step 1: Generate CFD Data**
```bash
python xfoil_runner.py
```
Outputs: `sweep_results.csv` containing aerodynamic coefficients at all operating conditions

**Step 2: Train Surrogate Model**
```bash
python surrogate_model.py
```
Outputs: `surrogate_surface.png` showing Cl, Cd, and L/D response surfaces with validation points

**Step 3: Run Optimization**
```bash
python optimizer.py
```
Outputs: `optimizer_result.png` showing L/D contour map with optimal point and performance curve

## Project Structure

```
├── xfoil_runner.py          # Automated XFOIL parameter sweep
├── surrogate_model.py        # GP training and validation
├── optimizer.py              # Surrogate-based L/D optimizer
├── sweep_results.csv         # CFD output dataset (13 runs)
├── surrogate_surface.png     # Surrogate validation plots
└── optimizer_result.png      # Optimization result visualization
```

## Future Work

- **OpenFOAM Validation**: Cross-validate XFOIL results with full Navier-Stokes simulation for high-fidelity verification
- **Custom NACA Generator**: Integrate parametric NACA airfoil geometry code to enable optimization over airfoil shape (thickness, camber) in addition to operating conditions
- **ParaView Visualization**: 3D flow field visualization of pressure and velocity distributions at optimal condition
