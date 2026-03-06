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

## Tools

- XFOIL 6.99
- Python 3.14
- pandas

## Next Steps

Gaussian Process surrogate model training using GPy (in progress)
