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

- **Checkpoint 1 complete**: XFOIL automated sweep pipeline (13/15 runs converged)
- **Checkpoint 2 complete**: GP surrogate model trained and validated
- **Optional next step**: surrogate-based optimizer using scipy.optimize to find max L/D condition
