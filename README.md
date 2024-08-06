# ezcascades
LAMMPS Python script for simulating high-dose irradiation damage for a given recoil spectrum

This script drives the [LAMMPS Molecular Dynamics Simulator](https://github.com/lammps/lammps) to simulate the evolution of microstructure under the influence of irradiation, simulated in the form of highly energetic atomic recoils. The script supports the simulation of single element materials and alloys, starting from the pristine crystal or a supplied structure, at thermal or athermal conditions, under displacement or stress constraints. The script enables simulation of large-scale high-dose microstructures (see [10.1103/PhysRevMaterials.6.063601](https://doi.org/10.1103/PhysRevMaterials.6.063601) for 21 mio atoms of tungsten at 1 dpa) through initialisation of multiple non-overlapping recoils per cascade iteration, with each iteration propagating the dose by a fixed dose increment. The repository comes with a script to convert [SRIM](http://www.srim.org/) electronic stopping tables to the format supported by LAMMPS.

The script requires LAMMPS to be built with the EXTRA-FIX and MANYBODY packages, for electronic stopping and embedded atom model potentials. LAMMPS must be compiled as a shared library linked to Python 3+ with the numpy, scipy, and mpi4py packages. See the [LAMMPS documentation](https://docs.lammps.org/Python_head.html) for more information on how to do so.

Recoil energies are drawn from a given recoil energy file. For example, the file can contain just a single line to simulate mono-energetic recoil energies (see [https://doi.org/10.1038/s41598-022-27087-w](10.1038/s41598-022-27087-w), or many raws with energies extracted from SRIM's COLLISIONS.TXT (see [https://doi.org/10.48550/arXiv.2401.13385](10.48550/arXiv.2401.13385)). The recoils are initialised such that they do not overlap spatially, thereby avoiding spurious coincidental recoil events.

The script writes regular log and restart files from which the simulation can be restarted simply by executing it again. The `json/*.json` files contains materials parameters, simulation settings, and paths to important files and directories, such as the EAM potential file and the scratch directory. Update the paths as appropriate for your system. The `json` file acts as the input file for the simulation, enabling the running of multiple similar simulations without having to manually edit the Python script. To start an example overlapping cascade simulation in tungsten from the terminal, use the script as follows:

```
mpirun -n 8 python3 ezcascades.py json/example_tungsten.json
```

As the simulation runs, the script logs some output to log/example_iron.log (iteration, dose, pxx, pyy, ..., lx, ly, lz), and writes dump files every few iterations into the scratch directory. The simulation can be stopped and restarted from the last snapshot any time. 

A sample job submission file for the CSD3 system is given in `jobs/data_initial.job`. This job runs a 1 million atom cascade simulation from an initial configuration (unzip `initial/data.perf10shear.zip` first).
