# ezcascades
LAMMPS Python script for simulating high-dose irradiation damage

Full details of the method are available at:
* Simultaneous cascades for large-scale high dose simulation: [10.1103/PhysRevMaterials.6.063601](https://doi.org/10.1103/PhysRevMaterials.6.063601)
* Varying recoil energies in different materials:             [10.1038/s41598-022-27087-w](https://doi.org/10.1038/s41598-022-27087-w)
* Cascades with recoil energies drawn from a spectrum:        [10.48550/arXiv.2401.13385](https://doi.org/10.48550/arXiv.2401.13385)


## Features

This script drives the [LAMMPS Molecular Dynamics Simulator](https://github.com/lammps/lammps) to simulate the evolution of microstructure under the influence of irradiation, simulated in the form of highly energetic atomic recoils. The script supports the simulation of single element materials and alloys, starting from the pristine crystal or a supplied structure, at thermal or athermal conditions, under displacement or stress constraints. The script enables simulation of large-scale high-dose microstructures through initialisation of multiple non-overlapping recoils per cascade iteration, with each iteration propagating the dose by a fixed dose increment. The repository comes with a script to convert [SRIM](http://www.srim.org/) electronic stopping tables to the format supported by LAMMPS.

The script requires LAMMPS to be built with the EXTRA-FIX and MANYBODY packages, for electronic stopping and embedded atom model potentials. LAMMPS must be compiled as a shared library linked to Python 3+ with the numpy, scipy, and mpi4py packages. See the [LAMMPS documentation](https://docs.lammps.org/Python_head.html) for more information on how to set this up.

Recoil energies are drawn from a given recoil energy file. For example, the file can contain just a single line to simulate mono-energetic recoil energies, or many rows with energies extracted from SRIM's `COLLISIONS.TXT`. The recoils are initialised such that their cascade heat spikes are unlikely to overlap spatially, thereby avoiding spurious coincidental recoil events.

The script writes regular log and restart files from which the simulation can be restarted simply by executing it again. The `json/*.json` files contains materials parameters, simulation settings, and paths to important files and directories, such as the EAM potential file and the scratch directory. Update the paths as appropriate for your system. The `json` file acts as the input file for the simulation, enabling the running of multiple similar simulations without having to manually edit the Python script.

## Running the code

To start an example overlapping cascade simulation in tungsten from the terminal, first ensure that a tungsten EAM potential is available, such as `W_MNB_JPCM17.eam.fs` by [Mason et al. (2017)](https://doi.org/10.1088/1361-648X/aa9776) available at the [NIST Interatomic Potentials repository](https://www.ctcms.nist.gov/potentials/), and then run the script as follows:

```
mpirun -n 8 python3 ezcascades.py json/example_tungsten.json
```

As the simulation runs, the script logs some output to `log/example_tungsten.log` (iteration, dose, pxx, pyy, ..., lx, ly, lz), and writes dump files every few iterations into the scratch directory. When the `simulation_clear` flag is set to zero in `json/example_tungsten.json`, the simulation can be stopped and restarted from the last snapshot. 

A sample job submission file for the CSD3 system is given in `jobs/data_initial.job`. This job runs a 1 million atom cascade simulation from an initial configuration (unzip `initial/data.perf10shear.zip` first).

## Input parameters

The json input file currently supports the following flags:

### Paths
* `job_name`: Name of this simulation (string), used as the prefix for exported data and log files, as well as the name of the directory containing exported dump files in the scratch directory. This should be a unique name for every simulation within a given simulation and scratch directory, otherwise data will be overwritten. [Example setting: `"test"`]
* `potential_path`: Path to directory containing the interatomic potential file (string). Useful when the potentials are stored in a central directory on the given system. [Example setting: `"~/Potentials/"`]
* `potential`: File name of the interatomic potential to be used for this simualation (string). The potential file should be contained in the `potential_path` directory. [Example setting: `"W_MNB_JPCM17.eam.fs"`.]
* `scratch_dir`: Path to directory where simulation snapshots can be stored (string). Here, a directory named `job_name` will be created, in which the simulation dump and restart files are stored. On an HPC system, this would typically be the scratch directory. [Example setting: `"/scratch/"`]
* `sim_dir`: Path to simulation directory (string). Paths to log files, stopping power files, and pka files, are typically given relative to this directory. This is typically the directory containing this repository. [Example setting: `"~/git/ezcascades/"`]
* `initialfile`: Path to LAMMPS data or dump file to be used as the initial structure (string). If this is set as `0` or `[]`, the system will be initialised as a single crystal according to the simulation cell settings below.  [Example settings: `"initial/data.perf10shear.data"`, or `0`]

### Atom properties
* `composition`: Atomic composition of the system (Python dictionary format), useful for creating random alloys. If an alloy is used, make sure that the element indices in this dictionary are consistent with the order of elements in the interatomic potential file. [Example settings: `{"1":  1.0}` for monoatomic system, or `{"1": 0.094, "2": 0.714, "3": 0.192}` for random alloy with three elements]
* `masses`: Atomic masses of the elements appearing in the system (Python dictionary format). The element indices must be consistent with the interatomic potential file. [Example settings: `{"1":  183.84}` for monoatomic system, or `{"1": 58.6934, "2": 55.845, "3": 51.9961}` for random alloy with three elements]
* `znums`: Charge number of the elements appearing in the system (Python dictionary format).  The element indices must be consistent with the interatomic potential file. [Example settings: `{"1":  74}` for monoatomic system, or `{"1": 28, "2": 26, "3": 24}` for random alloy with three elements]

### Data export settings
* `write_data`: Flag whether simulations snapshots are to be exported in regular intervals (boolean integer). The dump files are exported in `scratch_dir`/`job_name`. [Example setting: `0`, or `1`]
* `simulation_clear`: Flag whether the simulation should begin with clearing all previous output files (boolean integer). If yes, this will delete all files in `scratch_dir`/`job_name`, clear the log files `sim_dir`/log/`job_name`.log and `sim_dir`/log/`job_name`.pka, and the simulation will essentially begin fresh. If not, the script will attempt to resume from the most recent snapshot. [Example settings: `0` or `1`]
* `export_nth`: Export a dump snapshot every this many cascade iterations (integer). The snapshot will be exported as file "`scratch_dir`/`job_name`/`job_name`.*.dump", where "*" refers to the cascade iteration. The approximate dose in units of dpa can be calculated by the prduct of the cascade iteration and `incrementdpa`. If this value is zero, no snapshots are exported. [Example setting: `5`]

### Relaxation settings
* boxstress`: Stress constraints (in GPa) on the simulation box (Python dictionary). If `temperature` = 0, then the stress is maintained by the method of conjugate gradients after each cascade iteraiton. If `temperature` > 0.0, then the stress is maintained at all times with a Nose-Hoover barostat. The stress components are represented in LAMMPS format, with the flags "x", "y", "z", "xy", "xz", "yz". If shear components are entered, then the simulation box will changed to become triclinic. [Example setting: `{"x": 1.0, "xy": -0.2}`]
* `runCG`: Flag whether system is to be relaxed to a local energy minimum after each cascade iteration (boolean integer). If yes, then after completion of a cascade iteration, atomic velocities are zeroed and then atomic coordinates relaxed using the method of conjugate gradients. Afterwards, depending on the stress constraints on the simulation box (see `boxstress`), another relaxation might be performed, this time also relaxing box dimensions to reach any given stress constraints. This flag should not be used if `temperature` is larger than zero! [Example settings: `0` or `1`]
* `max_steps`: Maximum number of conjugate gradient relaxation steps (integer). [Example setting: `500000`]
* `etol`: Relative energy convergence criterion for conjugate gradient relaxation (string). [Example setting: `"1e-12"`]

### Recoil spectrum settings
* `PKAfile`: Path to file (relative to `sim_dir`) containing a set of recoil energies (in eV) representative of the irradiation (string). Recoil energies for ion irradiation of a given material can, for example, be exported from SRIM's `COLLISIONS.txt` file. For monoenergetic recoils, the file can contain a single line with the specified recoil energy. Cascade recoil energies are randomly drawn from the entries in this file until the specified dose increment (`incrementdpa`) is reached. [Example setting: `"./pkafiles/100.0eV.pka"`]
* `PKAmin`: Minimum recoil energy (in eV) to consider (float). Any drawn recoil energy below this value will be discarded. This value should be close to the material-dependent minimum displacement threshold energy. This is used to avoid simulating sub-threshold damage events, which are typically far more common than damage-generating events. If sub-threshold events are of interest, be aware that the notion of dose in dpa is not particularly well defined at low displacement energies, and the reported dose in dpa may not reflect a genuine quantity of interest. [Example setting: `50.0`]
* `PKAmax`: Maximum recoil energy (in eV) to consider (float). Any drawn recoil energy above this value will be discarded. This value should be close to the material-dependent cascade fragmentation. This is used to avoid simulating the very rare but highly energetic MeV+ cascades that do not fit into the simulation box. [Example setting: `100000.00`]
* `edavg`: Average displacement threshold energy (in eV) for computing the NRT dose (float). This value should be adjusted depending on the material. [Example setting: `90.0`]

### Electronic stopping model
* `stoppingfile`: Path to file (relative to `sim_dir`) containing the electronic stopping table (string). These can be generated from SRIM`s stopping tables using the `make_stopping_table.py` script. [Example setting: `"./stoppingfiles/W-W.dat"`]
* `electronic_stopping_threshold`: Minimum kinetic energy (in eV) above which electronic stopping is applied (float). [Example setting: `10.0`]
  
### Irradiation dose settings
* `maxdpa`: Maximum dose (in dpa) up to which the simulation is propagated to (float). Cascade iterations are repeated until the specified dose is reached. If the simulation terminates before the dose is reached, the simulation can be started again to continue from its most recent snapshot (make sure you set `simulation_clear`: 0!) [Example setting: `1.0`]
* `incrementdpa`: Target dose increment (in dpa) per cascade iteration (float). In each cascade iteration, recoil energies are randomly drawn from the `PKAfile` until the specified dose increment is reached, within a tolerance. These recoil energies will then be applied to random atoms in the system, spaced apart sufficiently far to not overlap. A too high dose increment will lead to significant heating of the system. [Example setting: `0.0002`]

### Thermostat and Barostat settings
* `temperature`: Thermostat temperature (in Kelvin) at which the simulation is maintained (float). The thermostat is a Langevin thermostat, acting on atoms with a temperature below `melting_temperature`. This is to avoid applying the damping term, which is essentially an [approximate representation of electron-phonon coupling in the low ionic velocity limit](https://doi.org10.1088/0953-8984/27/14/145401), to atoms that are likely to be in a molten environment or part of a collision cascade in the ballistic phase. [Example setting: `300.0`]
* `thermalisation_time` : Thermalisation time (in ps) for the system (float). If `temperature` > 0.0, then the system will be thermalised before the very first cascade iteration. This time should be long enough to let the system equilibrate.  [Example setting: `50.0`]
* `melting_temperature`: Approximate melting temperature (in Kelvin) of the material (float). Atoms with temperature above `melting_temperature` are not subjected to the Langevin thermostat. This temperature is also used to estimate the size of the cascade heat spike, which is used to ensure that atomic recoils are not placed too close together. [Example setting: `2000.0`]
* `langevin_damping_time`: Dampening time (in ps) for the Langevin thermostat (float). This number should be chosen to appoximately represent the [electron-phonon coupling timescale](https://doi.org10.1088/0953-8984/27/14/145401).  [Example setting: `5.0`]
* `barostat_damping_time`: Dampening time (in ps) for the Nose-Hoover thermostat (float). In thermal simulations (`temperature` > 0.0), the barostat is used to maintain the simulation box stress constraints. [Example setting: `50.0`]

### Cascade iteration competion criteria 
* `maxtemperature`: Maximum system temperature (in Kelvin) before the cascade iteration can be terminated (float). This parameter can be used to ensure that the system cools down to a specified maximum temperature before the next cascade iteration. This number should not be lower than `temperature`, otherwise the cascade iteration is unlikely to complete. Set this number to a very large value to strictly progress after a set amount of propagation time. Only when the cascade propagation time exceeds `minruntime` and the system temperature subceeds `maxtemperature`, will the next cascade iteration be initialised. [Example setting: `10000000.0`]
* `minruntime`: Maximum time (in ps) that each cascade iteration is propagated for (float). This parameter is used to ensure that each cascade iteration progresses for a set amount of time (with a very minor variability due to the use an adaptive time-step). Only when the cascade propagation time exceeds `minruntime` and the system temperature subceeds `maxtemperature`, will the next cascade iteration be initialised. [Example setting: `10.0`]

### Simulation cell settings 
The following settings are used to intialise the single crystal if no `initialfile` is supplied. The vectors given in `ix`, `iy` and `iz` should form a right-handed basis set. The simulation box is then spanned by the vectors `nx`\*`ix`, `ny`\*`iy`, and `nz`\*`iz`. If the EAM potential specifies an HCP structure, then an orthogonal simulation cell is built.
* `nx`: Number of repeats of `ix` lattice vectors (float). [Example setting: `100`]
* `ny`: Number of repeats of `iy` lattice vectors (float). [Example setting: `100`]
* `nz`: Number of repeats of `iz` lattice vectors (float). [Example setting: `100`]
* `ix`: First lattice vector (lattice units) spanning the simulation box (3x1 list of floats). [Example setting: `[1,0,0]`]
* `iy`: Second lattice vector (lattice units) spanning the simulation box (3x1 list of floats). [Example setting: `[0,1,0]`]
* `iz`: Third lattice vector (lattice units) spanning the simulation box (3x1 list of floats). [Example setting: `[0,0,1]`]

