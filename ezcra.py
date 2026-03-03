# initialise
import os
import sys
import json
import glob
import time 
import numpy as np

from lib.eam_info import eam_info
from lib.lindhard import Lindhard, quickdamage
from lib.helperfuncs import sample_spherical, get_dump_frame, is_triclinic 

# load lammps module and style and variable types
from lammps import lammps


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



def announce(string):
    mpiprint ()
    mpiprint ("=================================================")
    mpiprint (string)
    mpiprint ("=================================================")
    mpiprint ()
    return 0 


kB = 8.617333262e-5


def main():
    program_descripton = f'''
        LAMMPS Simulation script for running creation relaxation algorithm 

        Samanyu Tirumala, Mar 2026
        Max Boleininger, Aug 2024
        max.boleininger@ukaea.uk

        Licensed under the Creative Commons Zero v1.0 Universal
        https://creativecommons.org/publicdomain/zero/1.0/

        Distributed on an "AS IS" basis without warranties
        or conditions of any kind, either express or implied.

        USAGE:
        '''

    # -------------------
    #  IMPORT PARAMETERS    
    # -------------------
   
    inputfile = sys.argv[1]
    assert os.path.isfile(inputfile), "Error: input file %s not found." % inputfile  

    if (me == 0):
        with open(inputfile) as fp:
            all_input = json.loads(fp.read())
    else:
        all_input = None
    comm.barrier()

    # broadcast imported data to all cores
    all_input = comm.bcast(all_input, root=0)

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
    
    mpiprint ("Parameter input file %s:\n" % inputfile)
    for key in all_input:
        mpiprint ("    %s: %s" % (key, all_input[key]))
    mpiprint()

    # If new run, clear previous relaxation files
    # else, look for restart file
    timestamp = None
    restartfile = None
    if me == 0:
        if all_input['simulation_clear'] == 1:
            for file in glob.glob("%s/%s/*.dump" % (scrdir, job_name)):
                os.remove(file)
            for file in glob.glob("%s/%s/*.restart" % (scrdir, job_name)):
                os.remove(file)
            for file in glob.glob("%s/log/%s.log" % (simdir, job_name)):
                os.remove(file)
        else:
            # fetch restart file
            restartpath = "%s/%s/%s.restart" % (scrdir, job_name, job_name)
            if os.path.exists(restartpath):
                announce ("Found restart file: %s" % restartpath)
            else:
                announce ("Casacade restart file %s not found. Starting new simulation." % restartpath)
                restartpath = None
            restartfile = restartpath
            
        if not os.path.exists("%s/%s" % (scrdir, job_name)):
            os.mkdir("%s/%s" % (scrdir, job_name))

    timestamp = comm.bcast(timestamp, root=0)
    restartfile = comm.bcast(restartfile, root=0)

    # -------------------
    #  INPUT POTENTIAL 
    # ------------------- 

    potfile = potdir + potname

    # Any EAM potential file will be scraped for lattice parameters etc
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
    znums = all_input["znums"]


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


    # lattice vector norms
    sx = np.linalg.norm(ix)
    sy = np.linalg.norm(iy)
    sz = np.linalg.norm(iz)

    etol = float(all_input["etol"])
    etolstring = "%.5e" % etol

    # export every nth number of iterations
    export_nth = int(all_input["export_nth"])

    # maintain stresses during relaxation or during MD using barostat 
    if "boxstress" in all_input:
        boxstress = all_input["boxstress"]
    else:
        boxstress = {} 
    for sij in boxstress:
        boxstress[sij] = -boxstress[sij]*1e4 # convert GPa to bar

    # allow for the possibility of running from another starting point
    if "initial" in all_input:
        initial = all_input["initial"]
        initialtype = all_input["initialtype"]
    else:
        initial = None

    # whether to run CG after each cascade propagation step
    # CRA: by definition athermal, with CG after each propagation
    athermal = True
    runCG = True

    # max (canonical) dpa to propagate simulations for 
    if "maxdpa" in all_input:
        maxdpa = all_input["maxdpa"]
    else:
        maxdpa = 1.0

    # target (canonical) dpa increment per cra iteration, e.g 0.002 corresponds to insertion of 0.2% appm Frenkel pairs
    if "incrementdpa" in all_input:
        incrementdpa = all_input["incrementdpa"]
    else:
        incrementdpa = 0.002

    # CRA insertion exclusion radius in Angstrom to avoid inserting atoms into ill-defined regions of the potential 
    if "exclusion_radius" in all_input:
        exc_radius = all_input["exclusion_radius"]
    else:
        exc_radius = 0.2

    # fetch atomic composition
    composition = {int(_type):all_input["composition"][_type] for _type in all_input["composition"].keys()}

    # we do not consider elements with zero contribution to the composition.
    # this filtering out is done to not load extraneous species, which would
    # otherwise also require the stopping file to contain info on this species
    present_species = [str(_key) for _key in composition if composition[_key] != 0.0]
    nelements = len(present_species)

    masses = {_key:masses[_key] for _key in present_species}
    znums = {_key:znums[_key] for _key in present_species}

    composition = {int(_key):composition[int(_key)] for _key in present_species}

    potential.ele = [potential.ele[int(i)-1] for i in present_species]


    # ----------------------------
    #  DETERMINE CELL DIMENSIONS    
    # ----------------------------

    # Check for right-handedness in basis
    if np.r_[ix].dot(np.cross(np.r_[iy],np.r_[iz])) < 0:
        mpiprint ("Left Handed Basis!\n\n y -> -y:\t",iy,"->",)
        for i in range(3):
            iy[i] *= -1
        mpiprint (iy,"\n\n")


    # Start LAMMPS instance
    lmp = lammps()

    lmp.command('# Lammps input file')
    lmp.command('units metal')
    lmp.command('atom_style atomic')
    lmp.command('atom_modify map array sort 0 0.0')
    lmp.command('boundary p p p')
   
    # initialise lattice
    if potential.crystal == 'hcp':
        # hcp crystals are initialised such that box dimensions are orthogonal
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


    # read restart file and continue simulation from there, if available 
    if restartfile:
        announce("Restarting from last cascade file: %s" % restartfile)
        lmp.command('read_restart %s' % restartfile)

        # import log file and fetch last dose
        logdata = np.loadtxt('%s/log/%s.log' % (simdir, job_name))
        dpadose = logdata[-1,1]
        iteration = logdata[-1,0]
        
        lmp.command("print '# restart' append %s/log/%s.log" % (simdir, job_name))

    elif initial:
        # otherwise look for an initial file
        announce("Initialising structure from file: %s" % initial)

        # if neither initial nor restartfile have been given, initiate single crystal
        lmp.command('create_box %d r_simbox' % nelements) 

        if initialtype == "data":
            lmp.command('read_data %s' % initial)
        elif initialtype == "dump":
            initialframe = get_dump_frame(initial)

            tri, flag = is_triclinic (initial)
            if flag:
                announce (flag)
            if tri:
                lmp.command('run 0')
                lmp.command('change_box all triclinic')
            lmp.command('read_dump %s %d x y z purge yes add yes box yes replace no' % (initial, initialframe))
        
        lmp.command('reset_timestep 0')

    else:
        # if neither initial nor restartfile have been given, initiate single crystal
        lmp.command('create_box %d r_simbox' % nelements) 

        # initialise atoms as most commonly occuring species in the composition
        commontype = np.argmax(list(composition.values()))
        announce ("Initialising all atoms to the largest type: %d" % (1+commontype))
        lmp.command('create_atoms %d region r_simbox' % (1+commontype))
        natoms = lmp.extract_global("natoms", 0)

        if (me == 0):
            composition_array = np.random.choice(np.r_[:nelements], size=natoms, p=list(composition.values()))
        else:
            composition_array = None
        composition_array = comm.bcast (composition_array, root=0)

        # then set the remaining atomic species
        for _type in range(nelements):
            if _type == commontype:
                continue
            indices = tuple(1 + np.where(composition_array == _type)[0])
            announce ("Changing %d atoms to type: %d" % (len(indices), 1+_type))

            # work in batches, not setting too large groups a time
            maxgroupsize = 10000
            ngroups = int(len(indices)/maxgroupsize + 1)
            c = 0
            for _subindices in np.array_split(indices, ngroups):
                if len(_subindices) == 0:
                    continue
                mpiprint ("Batch %d out of %d..." % (c, ngroups))
                lmp.command('group gtype id' + " %d"*len(_subindices) % tuple(_subindices))
                lmp.command('set group gtype type %d' % (1+_type))
                lmp.command('group gtype delete')
                c += 1

    if not restartfile:
        dpadose = 0.0
        iteration = 0

    # load potential
    pottype = potfile.split('.')[-1]
    lmp.command('pair_style eam/%s' % pottype)
    lmp.command(('pair_coeff * * %s ' % potfile) + '%s '*nelements % tuple(potential.ele))

    # overwrite default masses
    for _i,_m in enumerate(masses.values()):
        lmp.command('mass %d %f' % (1+_i, _m)) 

    lmp.command('neighbor 3.0 bin')

    lmp.command('run 0')
    
    # thermo_style rate
    nth = 500
    lmp.command('thermo %d' % nth)
    lmp.command('thermo_style custom step press pe pxx pyy pzz pxy pxz pyz lx ly lz')
    lmp.command("thermo_modify line one format line '%8d %11.3e %15.8e %10.2e %10.2e %10.2e %10.2e %10.2e %10.2e %7.3f %7.3f %7.3f'")

    # if the simulation is not continuing from a restart file, relax structure and box dimensions
    if not restartfile:
        lmp.command('minimize %s 0 10000 10000' % (etolstring))

        rxstate = None
        if np.setdiff1d (["x","y","z","xy","xz","yz"], list(boxstress.keys())).size == 0 and np.sum(np.abs(list(boxstress.values()))) == 1e-9:
            # set as triclinic relaxation if all dimensions can relax
            lmp.command('fix ftri all box/relax tri 0.0 vmax 0.0001 nreset 100')
            rxstate = "tri"
        elif np.setdiff1d (["x","y","z"], list(boxstress.keys())).size == 0 and np.sum(np.abs(list(boxstress.values()))) < 1e-9:
            # set as orthorhombic relaxation if x,y,z dimensions can relax
            lmp.command('fix faniso all box/relax aniso 0.0 vmax 0.0001 nreset 100')
            rxstate = "aniso"
        else:
            # otherwise introduce multiple fixes
            for sij in boxstress:
                lmp.command('fix f%sfree all box/relax %s %f vmax 0.0001 nreset 100' % (sij, sij, boxstress[sij]))
            rxstate = "mixed"

        lmp.command('min_modify line quadratic')
        lmp.command('minimize %s 0 10000 10000' % (etolstring))

        # freeze box dimensions again
        if rxstate == "tri":
            lmp.command('unfix ftri') 
        elif rxstate == "aniso":
            lmp.command('unfix faniso')
        else:
            for sij in boxstress:
                lmp.command('unfix f%sfree' % sij)

        # wrap atoms back into the box
        lmp.command('reset_timestep 0')
        lmp.command('run 0')

        # print initial thermo quantities in log file
        lmp.command("variable vpe equal pe")
        lmp.command("variable vpxx equal pxx")
        lmp.command("variable vpyy equal pyy")
        lmp.command("variable vpzz equal pzz")
        lmp.command("variable vpxy equal pxy")
        lmp.command("variable vpxz equal pxz")
        lmp.command("variable vpyz equal pyz")
        lmp.command("variable vlx equal lx")
        lmp.command("variable vly equal ly")
        lmp.command("variable vlz equal lz")
        lmp.command("print '%d %f ${vpe} ${vpxx} ${vpyy} ${vpzz} ${vpxy} ${vpxz} ${vpyz} ${vlx} ${vly} ${vlz}' append %s/log/%s.log" % (iteration, 0.0, simdir, job_name))

    # print out first dump
    if not restartfile:
        if all_input['write_data']:
            lmp.command('write_dump all custom %s/%s/%s.%d.dump id type x y z' % (scrdir, job_name, job_name, iteration))

    # run CRA iterations 
    for cloop in range(1, int(1e6)):
   
        if dpadose >= maxdpa:
            announce ("Finished simulation. Current dose is %8.3f cdpa, with target dose given by %8.3f cdpa." % (dpadose, maxdpa))
            break
     
        # fetch number of atoms to be displaced 
        natoms  = lmp.extract_global("natoms", 0)
        nfrenkel = int(natoms*incrementdpa)
        appdose = nfrenkel/natoms # actually applied dose in cdpa: account for rounding errors 
        sys.stdout.flush () 
        mpiprint (f"Initialising {nfrenkel} Frenkel pairs, leading to a dose increment (cdpa) of: {appdose}")

        # we need the IDs types of atoms so that we know which to delete/regenerate 
        ids = np.ctypeslib.as_array(lmp.gather_atoms("id", 0, 1))
        types = np.ctypeslib.as_array(lmp.gather_atoms("type", 0, 1))

        # select nfrenkel random atoms to be deleted
        rnd_selection = None
        if me == 0:
            rnd_selection = np.random.choice(np.r_[:natoms], size=nfrenkel, replace=False)
        rnd_selection = comm.bcast(rnd_selection, root=0)
       
        delete_ids = ids[rnd_selection]
        insert_types = types[rnd_selection]

        # delete chosen atoms (create the vacancies)
        lmp.command("group gdel id" + " %d"*nfrenkel % tuple(delete_ids)) 
        lmp.command("delete_atoms group gdel compress yes")
        lmp.command("group gdel delete")

        # insert atoms anew (create the interstitials)
        sclock = time.time()
        _insert_types, _insert_counts = np.unique(insert_types, return_counts=True)

        # new random seed
        _rng = None
        if (me == 0):
            _rng = np.random.randint(10000000)
        _rng = comm.bcast (_rng, root=0)

        # create new atoms
        for _type, _count in zip(_insert_types, _insert_counts): 
            lmp.command(f"create_atoms {_type} random {_count} {_rng} NULL overlap {exc_radius} maxtry 10000")

        sclock = time.time() - sclock
        sys.stdout.flush () 
        mpiprint ("\nInserting atoms finished in %8.4f seconds.\n" % sclock)

        dpadose += appdose
        
        # first, relax atomic coordinates only
        lmp.command('minimize %s 0 10000 10000' % (etolstring))

        # next, also relax box dimensions
        rxstate = None
        if np.setdiff1d (["x","y","z","xy","xz","yz"], list(boxstress.keys())).size == 0 and np.sum(np.abs(list(boxstress.values()))) == 1e-9:
            # set as triclinic relaxation if all dimensions can relax
            lmp.command('fix ftri all box/relax tri 0.0 vmax 0.0001 nreset 100')
            rxstate = "tri"
        elif np.setdiff1d (["x","y","z"], list(boxstress.keys())).size == 0 and np.sum(np.abs(list(boxstress.values()))) < 1e-9:
            # set as orthorhombic relaxation if x,y,z dimensions can relax
            lmp.command('fix faniso all box/relax aniso 0.0 vmax 0.0001 nreset 100')
            rxstate = "aniso"
        else:
            # otherwise introduce multiple fixes
            for sij in boxstress:
                lmp.command('fix f%sfree all box/relax %s %f vmax 0.0001 nreset 100' % (sij, sij, boxstress[sij]))
            rxstate = "mixed"

            lmp.command('min_modify line quadratic')
            lmp.command('minimize %s 0 10000 10000' % (etolstring))

            # freeze box dimensions again
            if rxstate == "tri":
                lmp.command('unfix ftri') 
            elif rxstate == "aniso":
                lmp.command('unfix faniso')
            else:
                for sij in boxstress:
                    lmp.command('unfix f%sfree' % sij)

        # wrap atoms back into the box
        lmp.command('run 0')

        # print thermo quantities in log file
        lmp.command("variable vpe equal pe")
        lmp.command("variable vpxx equal pxx")
        lmp.command("variable vpyy equal pyy")
        lmp.command("variable vpzz equal pzz")
        lmp.command("variable vpxy equal pxy")
        lmp.command("variable vpxz equal pxz")
        lmp.command("variable vpyz equal pyz")
        lmp.command("variable vlx equal lx")
        lmp.command("variable vly equal ly")
        lmp.command("variable vlz equal lz")
        lmp.command("print '%d %f ${vpe} ${vpxx} ${vpyy} ${vpzz} ${vpxy} ${vpxz} ${vpyz} ${vlx} ${vly} ${vlz}' append %s/log/%s.log" % (iteration+cloop, dpadose, simdir, job_name))

        # write restart file always
        dfile = "%s/%s/%s.restart" % (scrdir, job_name, job_name)
        announce("Writing restart file: %s" % dfile)
        lmp.command('write_restart %s' % dfile) 

        # write dump file every 'export_nth' steps
        if ((iteration + cloop) % export_nth) == 0:
            dfile = "%s/%s/%s.%d.dump" % (scrdir, job_name, job_name, iteration+cloop)
            announce("Writing dump file %s." % dfile)
            lmp.command('write_dump all custom %s id type x y z' % dfile)     

        comm.barrier()

    lmp.close()

    return 0


if __name__ == "__main__":
    main()

    if mode == 'MPI':
        MPI.Finalize()


