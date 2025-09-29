# Gross-Pitaevskii Equation Solver for Spinor Bose–Einstein Condensates

This repository contains Python and Bash scripts to simulate Bose–Einstein condensates (BECs) with multiple spin components by numerically solving the **Gross–Pitaevskii equation (GPE)**.  
Different geometries are implemented: 1D, 3D Cartesian, and 3D with axial symmetry.

---

## Repository Structure

### `gp_1d`
Simulation code for **one-dimensional systems**.

- **`create_files.sh`**  
  Bash script that generates copies of the Python scripts with user-specified parameters, ready to run simulations.  

- **`execute_simulation.py`**  
  Main driver script. Running this file launches the simulation and stores results in the `output/` directory.  

- **`parameters.py`**  
  Defines all physical and numerical parameters of the system, such as trap frequencies, spin–orbit coupling strength, and grid resolution.  

- **`system_functions.py`**  
  Contains the core numerical routines used to evolve the system in time.  

---

### `gp_3d`
Simulation code for **three-dimensional systems in Cartesian coordinates**.  
Structure and functionality mirror `gp_1d`, but extended to 3D.  

---

### `gp_3d_axial`
Simulation code for **three-dimensional systems with axial symmetry** (cylindrical reduction).  

- Includes the same scripts as `gp_1d` and `gp_3d`.  
- Additional file:  
  - **`discrete_hankel_transform.py`**  
    Implements the Fourier transform in cylindrical coordinates using a discrete Hankel transform.  

---

## Usage

1. Navigate to the desired folder (`gp_1d`, `gp_3d`, or `gp_3d_axial`).  
2. Edit `parameters.py` to set system parameters.  
3. Run `create_files.sh` to generate parameterized copies of the scripts.  
4. Launch the simulation with:
   ```bash
   python execute_simulation.py

