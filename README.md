# CFD-Based Airfoil Surrogate Modeling Pipeline

This project implements an automated XFOIL parameter sweep across 5 angles of attack (-2, 0, 4, 8, 12 degrees) and 3 Reynolds numbers (500000, 1000000, 2000000) for the NACA 2412 airfoil, producing Cl, Cd, and Cm data to train a Gaussian Process surrogate model.

## Sweep Parameters

- **Angles of Attack**: -2°, 0°, 4°, 8°, 12°
- **Reynolds Numbers**: 500,000, 1,000,000, 2,000,000
- **Total Runs**: 15 attempted, 13 converged

## Results

The sweep results are stored in `sweep_results.csv`. Note that 2 runs failed to converge:
- AoA = 0° at Re = 500,000
- AoA = 12° at Re = 2,000,000

These failures are due to known XFOIL boundary layer convergence limitations.

## Results Summary

- **Surrogate model**: Gaussian Process Regressor (scikit-learn) with Matern kernel (nu=2.5)
- **Training points**: 10 (3 held out for validation)
- **Cl RMSE on holdout**: 0.0184
- **Cd RMSE on holdout**: 0.0013
- **Key finding**: L/D surface peaks at approximately AoA=4-5deg and Re=2,000,000 indicating optimal aerodynamic efficiency at moderate angle of attack and high Reynolds number
- **Validation plot**: see `surrogate_surface.png`

## Tools

- XFOIL 6.99
- Python 3.14
- pandas
- scikit-learn

## Project Status
- Checkpoint 1 complete: XFOIL automated sweep pipeline (13/15 runs converged)
- Checkpoint 2 complete: GP surrogate model trained and validated (Cl RMSE=0.0184, Cd RMSE=0.0013)
- Checkpoint 3 complete: Surrogate-based optimizer found maximum L/D condition

## Optimization Result
- Optimal AoA: 4.654 degrees
- Optimal Reynolds number: 2,000,000
- Predicted maximum L/D: 115.754
- Predicted Cl: 0.7608
- Predicted Cd: 0.00657
- Method: scipy.optimize L-BFGS-B minimizing negative L/D, run from 6 starting points to avoid local minima
- Result is physically consistent with known NACA 2412 aerodynamic behaviour - peak efficiency occurs at moderate angle of attack where lift is substantial but drag has not yet accelerated toward stall

## File Structure
- xfoil_runner.py — automated XFOIL parameter sweep
- surrogate_model.py — Gaussian Process surrogate training and validation
- optimizer.py — surrogate-based optimizer for maximum L/D
- sweep_results.csv — raw XFOIL output dataset (13 runs)
- surrogate_surface.png — Cl, Cd, L/D surrogate surface plots with validation points
- optimizer_result.png — L/D contour map with optimal point and L/D vs AoA curve
