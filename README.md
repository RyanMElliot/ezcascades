# ezcascades
LAMMPS Python script for running multiple overlapping cascades for a given recoil spectrum

The script requires LAMMPS to be built with the EXTRA-FIX package (for electronic stopping), and then compiled as a shared library linked to Python 3+ with the numpy, scipy, and mpi4py packages. 

The script supports drawing recoil energies from a given recoil energy file, it initialises multiple cascades such that they do not overlap spatially, and it writes regular log files and restart files from which the simulation can be restarted simply by executing it again. The `json/*.json` files contains materials parameters, simulation settings, and paths to important files and directories, such as the EAM potential file and the scratch directory. Update the paths as appropriate for your system.

The `json` file acts as the input file for the simulation, enabling the running of multiple similar simulations without having to manually edit the Python script. To start an example overlapping cascade simulation in iron from the terminal, use the script as follows:

```
mpirun -n 8 python3 srim_simple.py json/example_iron.json
```

As the simulation runs, the script logs some output to log/example_iron.log (iteration, dose, pxx, pyy, ..., lx, ly, lz), and writes dump files every few iterations into the scratch directory. The simulation can be stopped and restarted from the last snapshot any time.

A sample job submission file is given in `jobs/data_initial.job`. This runs a 1 million atom cascade simulations from an initial configuration (unzip `initial/data.perf10shear.zip` first).
