# initialise
import os, sys, json, glob
import numpy as np

from lib.eam_info import eam_info

# load lammps module and style and variable types
from lammps import lammps
from lammps import LMP_STYLE_GLOBAL, LMP_TYPE_SCALAR

import argparse, textwrap

# template to replace MPI functionality for single threaded use
class MPI_to_serial():
    def bcast(self, *args, **kwargs):
        return args[0]
    def barrier(self):
        return 0

# try running in parallel, otherwise single thread
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    me = comm.Get_rank()
    nprocs = comm.Get_size()
    mode = 'MPI'
except:
    me = 0
    nprocs = 1
    comm = MPI_to_serial()
    mode = 'serial'

def mpiprint(*arg):
    if me == 0:
        print(*arg)
        sys.stdout.flush()
    return 0

def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec.T


def announce(string):
    mpiprint ()
    mpiprint ("=================================================")
    mpiprint (string)
    mpiprint ("=================================================")
    mpiprint ()
    return 0 

kB = 8.617333262e-5

# better text formatting for argparse
class RawFormatter(argparse.HelpFormatter):
    def _fill_text(self, text, width, indent):
        return "\n".join([textwrap.fill(line, width) for line in textwrap.indent(textwrap.dedent(text), indent).splitlines()])


def main():
    program_descripton = f'''
        LAMMPS Simulation script for obtaining single cascade melting statistics for alloy

        Max Boleininger, Nov 2024
        max.boleininger@ukaea.uk

        Licensed under the Creative Commons Zero v1.0 Universal
        https://creativecommons.org/publicdomain/zero/1.0/

        Distributed on an "AS IS" basis without warranties
        or conditions of any kind, either express or implied.

        USAGE:
        '''

    # parse all input arguments
    parser = argparse.ArgumentParser(description=program_descripton, formatter_class=RawFormatter)

    parser.add_argument("jsonfile", help="path to input json file.")

    # box dimensions 
    parser.add_argument("-bdim", "--boxdims", nargs=3, default=[12,12,12],
                        help="Number of unit cell repeats in x,y,z (3 integers: nx ny nz) (default: 12 12 12)")

    # recoil energy 
    parser.add_argument("-epka", "--recoilenergy", type=float, default=100.0,
                        help="Primary knock-on atom recoil energy in eV (1 float) (default: No transformation)")

    args = parser.parse_args()

    # -------------------
    #  IMPORT PARAMETERS    
    # -------------------

    if (me == 0):
        with open(args.jsonfile) as fp:
            all_input = json.loads(fp.read())
    else:
        all_input = None
    comm.barrier()

    # broadcast imported data to all cores
    all_input = comm.bcast(all_input, root=0)

    # overwrite input file arguments with command line arguments
    all_input["eVPKA"] = args.recoilenergy
    all_input["nx"] = int(args.boxdims[0])
    all_input["ny"] = int(args.boxdims[1])
    all_input["nz"] = int(args.boxdims[2])

    # -----------------------
    #  SET INPUT PARAMETERS 1   
    # -----------------------

    job_name = all_input['job_name']
    
    potdir  = all_input['potential_path']
    potname = all_input['potential']

    simdir = all_input["sim_dir"]
    scrdir = all_input["scratch_dir"]

    mpiprint ("Running in %s mode." % mode)
    mpiprint ("Job %s running on %s cores.\n" % (job_name, nprocs))
    
    mpiprint ("Parameter input file %s:\n" % args.jsonfile)
    for key in all_input:
        mpiprint ("    %s: %s" % (key, all_input[key]))
    mpiprint()

    # If new run, start from iteration 0 
    # else, look for last log file
    if me == 0:
        # fetch last log file
        logfiles = glob.glob("%s/log/%s.%.3feV.*.rec" % (simdir, job_name, all_input["eVPKA"]))
        if len(logfiles)>0:
            print ("Fetching last log file.")

            logindex = np.array([_l.split('.')[-2] for _l in logfiles], dtype=int)
            logindex = np.max(logindex)
            mpiprint ("Resuming simulation from iteration %d." % logindex)
        else:
            logindex = 0 

        if not os.path.exists("%s/%s" % (scrdir, job_name)):
            os.mkdir("%s/%s" % (scrdir, job_name))
    else:
        logindex = None

    logindex = comm.bcast(logindex, root=0)

    comm.barrier()


    # -------------------
    #  INPUT POTENTIAL 
    # ------------------- 

    potfile = potdir + potname

    # Any monatomic EAM potential file will be scraped for lattice parameters etc
    potential = eam_info(potfile) # Read off

    mpiprint ('''Potential and elastic constants information:

    Elements, %s,
    mass: %s,
    lattice: %s,
    crystal: %s,
    cutoff: %s
    ''' % (
        potential.ele, potential.mass, 
        potential.lattice, potential.crystal, 
        potential.cutoff)
    )

    alattice = potential.lattice
    masses = all_input["masses"]

    # -----------------------
    #  SET INPUT PARAMETERS 2   
    # -----------------------

    if potential.crystal == 'hcp':
        # if not supplied, initialise ideal c/a ratio for hcp lattice
        if "c_over_a" in all_input:
            c_over_a = all_input["c_over_a"]
        else:
            c_over_a = np.sqrt(8./3.)

        clattice = c_over_a * alattice 

        # LAMMPS lattice vectors for hcp lattice
        ix = np.r_[alattice, 0, 0]
        iy = np.r_[0, np.sqrt(3.)*alattice, 0]
        iz = np.r_[0, 0, clattice]

        # integer lattice repeats for simulation box size
        # here rescaled to give broadly similar dimensions for similar nx,ny,nz
        nx = np.round(all_input['nx'])
        ny = np.round(all_input['ny']/np.sqrt(3.))
        nz = np.round(all_input['nz']/np.sqrt(8./3.))

    else:
        # all INTEGER lattice vectors for LAMMPS lattice orientation
        ix = np.r_[all_input['ix']]
        iy = np.r_[all_input['iy']]
        iz = np.r_[all_input['iz']]

        nx = all_input['nx']
        ny = all_input['ny']
        nz = all_input['nz']

        c_over_a = 1.0

    # lattice vector norms
    sx = np.linalg.norm(ix)
    sy = np.linalg.norm(iy)
    sz = np.linalg.norm(iz)

    # box lengths free or not
    freebox = all_input["freebox"]
    etol = float(all_input["etol"])
    etolstring = "%.5e" % etol

    # maintain sigma_xx, yy and zz stresses during relaxation if free box lengths
    if "boxstress" in all_input:
        boxstress = -np.r_[all_input["boxstress"]]*1e4 # convert GPa to bar
    else:
        boxstress = [0., 0., 0.]

    # temperature in kelvin
    temp = all_input["temperature"]

    # ------------------------------
    #  COLLISION CASCADE PARAMETERS
    # ------------------------------

    # PKA energy (eV) 
    eVexcite = all_input["eVPKA"]
 
    # minimum electronic stopping energy (eV)
    electronic_stopping_threshold = all_input["electronic_stopping_threshold"]

    # path to LAMMPS electronic stopping file 
    stoppingfile = all_input["stoppingfile"]

    # minimum propagation time in ps (useful for small cascades) 
    minruntime = all_input["minruntime"]

    alloy = True
    if alloy:
        composition = {int(_type):all_input["composition"][_type] for _type in all_input["composition"].keys()}

    # ----------------------------
    #  DETERMINE CELL DIMENSIONS    
    # ----------------------------

    # Check for right-handedness in basis
    if np.r_[ix].dot(np.cross(np.r_[iy],np.r_[iz])) < 0:
        mpiprint ("Left Handed Basis!\n\n y -> -y:\t",iy,"->",)
        for i in range(3):
            iy[i] *= -1
        mpiprint (iy,"\n\n")

    maxruns = all_input["maxruns"]

    for _it in range(logindex+1, maxruns):

        # Start LAMMPS instance
        lmp = lammps()

        lmp.command('# Lammps input file')
        lmp.command('units metal')
        lmp.command('atom_style atomic')
        lmp.command('atom_modify map array sort 0 0.0')
        lmp.command('boundary p p p')
       
        # initialise lattice
        if potential.crystal == 'hcp':
            lmp.command('lattice custom 1.0 a1 %f %f %f a2 %f %f %f a3 %f %f %f basis 0.0 0.0 0.0 basis 0.5 0.5 0.0 basis 0.5 %f 0.5 basis 0.0 %f 0.5' % (
                ix[0], ix[1], ix[2],
                iy[0], iy[1], iy[2],
                iz[0], iz[1], iz[2],
                5./6., 1./3.))
        else:
            lmp.command('lattice %s %f orient x %d %d %d orient y %d %d %d orient z %d %d %d' % (
                        potential.crystal,
                        potential.lattice,
                        ix[0], ix[1], ix[2],
                        iy[0], iy[1], iy[2],
                        iz[0], iz[1], iz[2]))

        # cubic simulation cell region
        lmp.command('region r_simbox block 0 %d 0 %d 0 %d units lattice' % (nx, ny, nz))


        if (me == 0):
            rnglist = np.random.randint(1e8, size=100000)
        else:
            rnglist = None
        rnglist = comm.bcast(rnglist, root=0)
        comm.barrier()

        if alloy: 
            nelements = len(potential.ele)
            lmp.command('create_box %d r_simbox' % nelements) 
            lmp.command('create_atoms 1 region r_simbox')

            natoms = lmp.extract_global("natoms", 0)
            composition_array = np.random.choice(np.r_[:nelements], size=natoms, p=list(composition.values()))

            for _type in range(nelements):
                indices = tuple(1 + np.where(composition_array == _type)[0])
                lmp.command('group gtype id' + " %d"*len(indices) % indices)
                lmp.command('set group gtype type %d' % (1+_type))
                lmp.command('group gtype delete')
        else:
            lmp.command('create_box 1 r_simbox')
            lmp.command('create_atoms %d region r_simbox' % atype)

        # just a hack to handle alloys
        if alloy:
            lmp.command('pair_style eam/alloy')
            lmp.command(('pair_coeff * * %s ' % potfile) + '%s '*nelements % tuple(potential.ele))

            # overwrite default masses
            for _i,_m in enumerate(masses.values()):
                lmp.command('mass %d %f' % (1+_i, _m)) 
        else:
            #lmp.command('mass            1 %f' % mass)
            lmp.command('pair_style eam/fs')
            lmp.command('pair_coeff * * %s %s' % (potfile, ele))
        lmp.command('neighbor 3.0 bin')


        lmp.command('compute ccount all count/type atom')
        lmp.command('thermo_style custom step temp pe press c_ccount[1] c_ccount[2] c_ccount[3]') 

        lmp.command('run 0')
        
        # use finer resolution for small energies 
        if eVexcite <= 100.0:
            nth = 20
        else:
            nth = 50

        # time step in femtoseconds
        timestep = 0.002 
        lmp.command('timestep %f' % timestep)

        lmp.command('thermo %d' % nth)
        lmp.command('thermo_style custom step temp pe press') 
    
        # minimise box dimensions and atoms
        if len(freebox) > 0:
            lmap = {"x": 0, "y": 1, "z":2}
            for _vx in freebox:
                lmp.command('fix f%sfree all box/relax %s %f vmax 0.0005' % (_vx, _vx, boxstress[lmap[_vx]])) 
            lmp.command('minimize %s 0 10000 10000' % (etolstring))

            # fix box dimensions again
            for _vx in freebox:
                lmp.command('unfix f%sfree' % _vx) 
        else:
            lmp.command('minimize %s 0 10000 10000' % (etolstring))

        # wrap atoms back into the box
        lmp.command('reset_timestep 0')
        lmp.command('run 0')
 
        # if requested, print out first dump
        if all_input['write_data']:
            #lmp.command('write_dump all custom %s/%s/%s_%d.0.dump id type x y z' % (scrdir, job_name, job_name, _it))
            lmp.command('write_data %s.%.3feV.0.data' % (job_name, all_input["eVPKA"]))
       
        # turn on NVE ensemble
        lmp.command('fix fnve all nve')
        
        # initialise stopping model
        lmp.command('fix fstopping all electron/stopping %f %s' % (electronic_stopping_threshold, stoppingfile))

        # initialise velocities
        if temp > 0.0:
            rndnumber = np.random.randint(1e8)
            announce("Initialising random velocities with random number: %d" % rndnumber)
            lmp.command('velocity all create %f %d mom yes rot no' % (2*temp, rndnumber))
            # should probably add thermalisation here
            # ... 
        else:
            lmp.command('velocity all create 0.0 1 mom yes rot no')
        
        comm.barrier()
 
        # wrap atoms back into the box
        lmp.command('reset_timestep 0')
        lmp.command('run 0')
     
        # extract number of atoms
        natoms = lmp.extract_global("natoms", 0)
       
        # we need the IDs so that we know which atom type we are kicking 
        _ids = np.ctypeslib.as_array(lmp.gather_atoms("type", 0, 1))

        # select a random atom to apply the PKA energy
        if (me == 0):
            kickindex = np.random.randint(natoms) # +1 for LAMMPS
            _type = _ids[kickindex]
            # convert mass from Dalton to eV ps^2/Ang^2
            kickvelocity = np.sqrt(2.*eVexcite/(masses[str(_type)] * 1.03499e-4))
            vvec = kickvelocity*sample_spherical(1)[0]   
        else:
            kickindex = None
            vvec = None
        comm.barrier()
       
        kickindex = comm.bcast(kickindex, root=0)
        vvec = comm.bcast(vvec, root=0)

        # initialise parameters for processing the melt
        rcna = {}
        cnaid = {}
        ccoord = {}
        ccut = {}

        # values taken from https://docs.lammps.org/compute_cna_atom.html
        x = c_over_a/np.sqrt(8/3.)
        rcna['bcc'] = 1.207  * alattice
        rcna['fcc'] = 0.8536 * alattice
        rcna['hcp'] = 0.5*(1 + np.sqrt((4.+2*x*x)/3.)) * alattice

        cnaid['bcc'] = 3
        cnaid['fcc'] = 1
        cnaid['hcp'] = 2

        # incomplete coordination numbers for removing atoms neighbouring the melt
        # for comparision, complete numbers: bcc=8, fcc=12, hcp=12
        ccoord['bcc'] = 6
        ccoord['fcc'] = 8
        ccoord['hcp'] = 8

        # cutoff between 1st and 2nd neighbour for incomplete coordination number (except for bcc)
        ccut['bcc'] = 0.933 * alattice #* 1.016
        ccut['fcc'] = 0.853 * alattice
        ccut['hcp'] = 1.300 * alattice # c_over_a ratio should be larger than 1.3 obviously


        # apply random kick
        announce ("Kicking atom ID %d of type %d by %12.6f Ang/ps." % (1+kickindex, 1+_type, np.linalg.norm(vvec)))
        lmp.command('group gkick id %d' % (1+kickindex))
        lmp.command('velocity gkick set %f %f %f sum yes units box' % tuple(vvec))
        lmp.command('group gkick delete') 

        # compute for evaluating per-atom kinetic energy, as used for dynamical Langevin damping group
        lmp.command('compute cke all ke/atom')
        lmp.command('compute cketot all reduce sum c_cke')

        # create a dynamical group of low kinetic energy atoms for applying Langevin thermostat
        damping_cutoff = 3*kB*all_input["langevin_melting_temperature"]
        lmp.command('variable vekin atom "c_cke < %f"' % damping_cutoff)
        lmp.command('group gdamped dynamic all var vekin every %d' % nth)
        lmp.command('variable vndamped equal count(gdamped)')

        # define compute for CNA
        _rcna = rcna[potential.crystal]
        _cnaid = cnaid[potential.crystal]

        lmp.command('compute ccna all cna/atom %f' % _rcna)
        lmp.command('variable vmelt atom c_ccna!=%d' % _cnaid)
        lmp.command('variable vmeltp equal v_vmelt[1]')

        lmp.command('thermo_style custom step dt time temp pe ke v_vmeltp v_vndamped')
        lmp.command("thermo_modify line one format line '%8d %6e %10.6f %10.6f %15.8e %15.8e %3f %8.0f'")
        lmp.command('run 0')

        # define compute for coordination number
        _ccut = ccut[potential.crystal]
        _ccoord = ccoord[potential.crystal]

        lmp.command('group gmelt dynamic all var vmelt every %d' % nth)
        lmp.command('compute ccoord gmelt coord/atom cutoff %f group gmelt' % _ccut)
        lmp.command('variable vrec atom "c_ccoord>=%d"' % _ccoord)
        lmp.command('compute csumrec all reduce sum v_vrec')
        lmp.command('compute csummelt all reduce sum v_vmelt')

        mpiprint("Common Neb Analysis parameters: Rcut = %6.3f,   ID = %d" % (_rcna, _cnaid))
        mpiprint("Coordination number parameters: Rcut = %6.3f, nebs = %d" % (_ccut, _ccoord))

        lmp.command('thermo %d' % nth)
        lmp.command('thermo_style custom step dt time temp pe ke etotal c_csummelt c_csumrec v_vndamped')
        lmp.command("thermo_modify line one format line '%8d %6e %10.6f %10.6f %15.8e %15.8e %15.8e %5f %5f %8.0f'")
        lmp.command('run 0')

        lmp.command('fix ftimestep all dt/reset 1 NULL 0.002 0.02 units box')

        # apply Langevin damping to dynamic group
        lmp.command('run 0')

        rndnumber = None
        if (me == 0):
            rndnumber = np.random.randint(1e8)
        rndnumber = comm.bcast(rndnumber, root=0)
        
        tdamp = all_input["langevin_damping_time"]
        lmp.command('fix flangevin gdamped langevin %f %f %f %d' % (temp+1e-9, temp+1e-9, tdamp, rndnumber))

        maxloops = int(1e6)
        nsteps = nth
        announce("Running loops of %d steps until a runtime of %f picoseconds is reached." % (nsteps, minruntime))

        # simulation time value is not reset, so need to keep track of the difference 
        initial_time = float(lmp.get_thermo('time'))

        for k in range(maxloops):
            lmp.command('run %d' % nsteps)
            current_time = float(lmp.get_thermo('time'))

            # extract number of molten atoms from compute
            current_melt = int(lmp.extract_compute('csumrec', LMP_STYLE_GLOBAL, LMP_TYPE_SCALAR))
            lmp.command("print '%f %d' append %s/log/%s.%.3feV.%d.rec" % (current_time, current_melt, simdir, job_name, all_input["eVPKA"], _it))

            if (current_time-initial_time) > minruntime:
                mpiprint("Reached minimum run time, propagation complete.")
                break


        lmp.close()

    return 0


if __name__ == "__main__":
    main()

    if mode == 'MPI':
        MPI.Finalize()




