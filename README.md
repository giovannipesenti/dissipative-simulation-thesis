# Dissipative 3-SAT Quantum Circuit Simulations

This repository contains the Python simulation pipeline developed for my Bachelor's thesis project in Physics, focusing on dissipative quantum circuit simulations, quantum satisfiability (3-SAT) problem instances, and ground-state convergence dynamics.

## Overview

The script simulates the dissipative dynamics of quantum 3-SAT Hamiltonians under projective measurements and depolarizing noise channels. It evaluates:
- **Random 3-SAT Instance Generation**: Supporting both 1-variable and 2-variable overlapping structures between adjacent clauses.
- **Trajectory Simulations**: Tracking energy expectation values, single-step measurements, and convergence times to the ground state ($t_{gs}$).
- **Data Persistence**: Storing multi-run simulation arrays and aggregated statistics efficiently using HDF5 (`h5py`) with Gzip compression.

## Requirements

The simulation pipeline requires Python 3.x and the following scientific libraries:
- `numpy`
- `scipy`
- `matplotlib`
- `h5py`

You can install the required dependencies via pip:
```bash
pip install numpy scipy matplotlib h5py
