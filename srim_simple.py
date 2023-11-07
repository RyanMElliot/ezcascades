# initialise
import os, sys, json, glob
import time
import numpy as np

sys.path.insert(0,'../..')

from scipy.spatial import cKDTree 
from lammps import lammps

from lib.eaminfo import Import_eamfs
from lib.lindhard import Lindhard, nrtdamage

from ctypes import *

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


def get_dump_frame(fpath):
    # helper function for fetching frame in a dump file
    with open(fpath, 'r') as fp:
        fp.readline()
        frame = int(fp.readline())
    return frame

def is_triclinic(fpath):
    # helper function for determining if dump file describes an orthogonal or triclinic box
    with open(fpath, 'r') as fp:
        fp.readline()
        fp.readline()
        fp.readline()
        fp.readline()
        string = fp.readline()
        if "xy xz yz" in string:
            tri = True
        else:
            tri = False

        # also, check next line and see if atom types are included
        string = fp.readline()
        if " type " not in string:
            announce("Warning: no atomic type column found in %s. Command read_dump will fail!" % fpath)
    return tri            


def announce(string):
    mpiprint ()
    mpiprint ("=================================================")
    mpiprint (string)
    mpiprint ("=================================================")
    mpiprint ()
    return 0 


def tabulate_stopping(filename, massnumber, Tmelt, B0nuclear, B0electronic, Ecut, Efinal):

    mass = massnumber * 103.643 # in eV*fs^2/Ang^2
    kB = 0.0000861733 # in eV/K
    Emelt = 1.5*kB*Tmelt # in eV

    dE = 0.001
    E_nuc = np.arange(0, Emelt, dE)
    E_nuc = np.r_[E_nuc, Emelt+1e-6]
    E_nuc[0] = 1.e-9
    S_nuc = np.sqrt(2.0*E_nuc/mass) * B0nuclear

    dE = 0.1
    E_zero = np.arange(Emelt+2e-6, Ecut-2e-6, dE)
    E_zero = np.r_[E_zero, Ecut-1e-6]
    S_zero = 0.*E_zero

    dE = 1.0
    E_ele = np.arange(Ecut, Efinal, dE)
    E_ele = np.r_[E_ele, Efinal+1e-6]
    S_ele = np.sqrt(2.0*E_ele/mass) * B0electronic 

    with open(filename, 'w') as sfile:
        sfile.write("#       atom-1\n")
        sfile.write("# eV    eV/Ang     # units metal\n")

        # first, write nuclear stopping
        for i in range(len(E_nuc)):
            sfile.write("%16.12f %16.12f\n" % (E_nuc[i], S_nuc[i]))

        # next, write zero stopping for molten region
        for i in range(len(E_zero)):
            sfile.write("%16.12f %16.12f\n" % (E_zero[i], S_zero[i]))

        #sfile.write("%16.12f %16.12f\n" % (Emelt+2e-6, 0.0))
        #sfile.write("%16.12f %16.12f\n" % (Ecut-1e-6, 0.0))

        # finally, write electronic stopping
        for i in range(len(E_ele)):
            sfile.write("%16.12f %16.12f\n" % (E_ele[i], S_ele[i]))
    return 0


def main():
    mpiprint ('''
LAMMPS python script for running simultaneous overlapping cascades with a given recoil spectrum.

Max Boleininger 2023, max.boleininger@ukaea.uk 
    ''')

    # -------------------
    #  IMPORT PARAMETERS    
    # -------------------

    inputfile = sys.argv[1]
    
    if (me == 0):
        with open(inputfile) as fp:
            all_input = json.loads(fp.read())
    else:
        all_input = None
    comm.barrier()

    # broadcast imported data to all cores
    for nc in range(nprocs):
        if nc != 0:
            sys.stdout.flush()
            all_input = comm.bcast(all_input, root=0)
    comm.barrier()

    # -----------------------
    #  SET INPUT PARAMETERS 1   
    # -----------------------

    job_name = all_input['job_name']
    
    potdir  = all_input['potential_path']
    potname = all_input['potential']
    atype   = all_input['atomtype']

    # decrement atomic type by 1 as LAMMPS starts counting from 1, while Python from 0 
    atype -= 1

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
    if me == 0:
        timestamp = 0 
        restartfile = None

        if all_input['relax_clear'] == 1:
            for file in glob.glob("%s/%s/*.dat" % (scrdir, job_name)):
                os.remove(file)
            for file in glob.glob("%s/%s/*.data" % (scrdir, job_name)):
                os.remove(file)
            for file in glob.glob("%s/%s/*.dump" % (scrdir, job_name)):
                os.remove(file)
        else:
            # fetch restart file from restart dump file
            restartpath = "%s/%s/%s.restart" % (scrdir, job_name, job_name)
            if os.path.exists(restartpath):
                announce ("Restarting from file %s" % restartpath)
                # get timestamp
                restartfile = restartpath
                with open(restartfile, 'r') as rfile:
                    rfile.readline()    
                    timestamp = int(rfile.readline())
            else:
                announce ("Restart file %s not found. Starting new simulation." % restartpath)
                restartpath = None
            restartfile = restartpath
            
        if not os.path.exists("%s/%s" % (scrdir, job_name)):
            os.mkdir("%s/%s" % (scrdir, job_name))
    else:
        timestamp = None
        restartfile = None

    timestamp = comm.bcast(timestamp, root=0)
    restartfile = comm.bcast(restartfile, root=0)

    comm.barrier()

    # -------------------
    #  INPUT POTENTIAL    
    # ------------------- 

    potfile = potdir + potname
    elafile = potname.split('.')[-2] + '_constants.json'

    # Any EAM fs or alloy potential file will be scraped for lattice parameters etc
    if (me == 0): 
        potential = Import_eamfs(potfile)
    else:
        potential = None
    
    # broadcast imported data to all cores 
    potential = comm.bcast(potential, root=0)

    mpiprint ('''\nPotential information:

    Elements, %s,
    mass: %s,
    Z number: %s,
    lattice: %s,
    crystal: %s,
    cutoff: %s
    ''' % ( 
        tuple(potential.elements), tuple(potential.mass.values()), tuple(potential.znum.values()), 
        tuple(potential.alat.values()), tuple(potential.atyp.values()), 
        potential.cutoff)
    )   


    # fetch potential values for host lattice
    mass     = potential.mass[atype] # mass
    znum     = potential.znum[atype] # z number
    lattice  = potential.atyp[atype] # crystal structure
    alattice = potential.alat[atype] # lattice constant

    
    # -----------------------
    #  SET INPUT PARAMETERS 2   
    # -----------------------

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

    # box lengths free or not
    freebox = all_input["freebox"]

    etol = float(all_input["etol"])
    etolstring = "%.5e" % etol

    # export every nth number of iterations
    export_nth = int(all_input["export_nth"])

    # maintain sigma_xx, yy and zz stresses during relaxation if free box lengths
    if "boxstress" in all_input:
        boxstress = -np.r_[all_input["boxstress"]]*1e4 # convert GPa to bar
    else:
        boxstress = [0., 0., 0.]

    # allow for the possibility of running from another starting point
    if "initial" in all_input:
        initial = all_input["initial"]
        initialtype = all_input["initialtype"]
    else:
        initial = None

    # temperature damping parameter in metal time units (ps)
    if "temp_damping" in all_input:
        temp_damping = all_input["temp_damping"] 
    else:
        temp_damping = 5.0
 
    if "press_damping" in all_input:
        press_damping = all_input["press_damping"] 
    else:
        press_damping = 25.0
 
    # temperature in kelvin
    temp = all_input["temperature"]

    # ------------------------------
    #  COLLISION CASCADE PARAMETERS
    # ------------------------------

    # PKA events (eV) 
    PKAfile = all_input["PKAfile"] 

    # Import PKA events 
    if me == 0:
        mpiprint ("Importing PKA spectrum %s" % PKAfile)
        pkas = np.loadtxt(PKAfile, dtype=float)
        if pkas.shape == ():
            pkas = np.array([pkas])
        mpiprint ("Imported PKA spectrum of size %d events, min, mean, max energies: %6.4f %6.4f %6.4f" % (len(pkas), np.min(pkas), np.mean(pkas), np.max(pkas)))
        mpiprint ()
    else:
        pkas = None
    comm.barrier()
    pkas = comm.bcast(pkas, root=0)
    comm.barrier()

    # min threshold displacement energy and cascade fragmentation energy
    pkamin = all_input["PKAmin"]
    pkamax = all_input["PKAmax"]
    edavg = all_input["edavg"]

    # nuclear and electronic damping coefficients (eV*fs/Ang^2) 
    B0nuclear    = all_input['B0nuclear']
    B0electronic = all_input['B0electronic']

    # melting temperature
    Tmelt = all_input['Tmelt']

    # lower kinetic energy cutoff for electronic stopping (typically between 1 and 10 eV) 
    Ecut = all_input['Ecut']

    # the next cascade iteration does not proceed until the system has cooled below this temperature 
    tmax = all_input["maxtemperature"]

    # minimum propagation time in ps (useful for small cascades or to enforce a fixed dose-rate) 
    minruntime = all_input["minruntime"]

    # if 1, finish propagation with CG
    runCG = all_input["runCG"]

    # max dpa to propagate simulations for 
    if "maxdpa" in all_input:
        maxdpa = all_input["maxdpa"]
    else:
        maxdpa = 1.0

    if "incrementdpa" in all_input:
        incrementdpa = all_input["incrementdpa"]
    else:
        incrementdpa = 0.0002


    # maximum energy for tabulation, should be higher than the PKA energy
    Efinal = all_input['Efinal']
    if Efinal < pkamax:
        mpiprint("Warning: maximum energy for tabulation Efinal is lower than maximal PKA energy.")
        mpiprint("Efinal is now set to 2*(maximal PKA energy) to avoid crashing.")

        Efinal = 2*pkamax

    stoppingfile = "log/stopping_%s.dat" % job_name
    comm.barrier()

    # tabulate stopping power
    if me == 0:
        print ("\nTabulating stopping power in file %s.\n" % stoppingfile)
        tabulate_stopping(stoppingfile, mass, Tmelt, B0nuclear, B0electronic, Ecut, Efinal)
    comm.barrier()


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
    #lmp = lammps(name="mpi", cmdargs=["-sf", "gpu", "-pk", "gpu", "4", "neigh", "no", "binsize", "8"]):w
    lmp = lammps(name="intel_cpu_intelmpi", cmdargs=["-sf", "intel"])

    lmp.command('# Lammps input file')
    lmp.command('units metal')
    lmp.command('atom_style atomic')
    lmp.command('atom_modify map array sort 0 0.0')
    lmp.command('boundary p p p')
   
    # initialise lattice 
    lmp.command('lattice %s %f orient x %d %d %d orient y %d %d %d orient z %d %d %d' % (
                lattice,
                alattice,
                ix[0], ix[1], ix[2],
                iy[0], iy[1], iy[2],
                iz[0], iz[1], iz[2]))

    # cubic simulation cell region
    lmp.command('region r_simbox block 0 %d 0 %d 0 %d units lattice' % (nx, ny, nz))

    # is an alloy being simulated? 
    if 'alloy' in all_input:
        alloy = all_input['alloy'] 
        nelements = len(potential.elements)
    else:
        alloy = False
        nelements = 1

    # create simulation box and define masses only if starting from dump file
    if restartfile or (initial and initialtype == "dump"):
        lmp.command('create_box %d r_simbox' % nelements)

        if alloy:
            for _i,_m in enumerate(potential.mass.values()):
                lmp.command('mass %d %f' % (_i+1, _m)) 
        else:
            lmp.command('mass 1 %f' % list(potential.mass.values())[atype])

        lmp.command('create_atoms %d region r_simbox' % (atype+1))


    # read restart file and continue simulation from there, if available 
    if restartfile:
        announce("Restarting from file: %s" % restartfile)
        if is_triclinic(restartfile):
            lmp.command('run 0')
            lmp.command('change_box all triclinic')

        lmp.command('read_dump %s %d x y z purge yes add yes box yes replace no' % (restartfile, timestamp))
        lmp.command('reset_timestep 0')
        lmp.command("print '# restart' append %s/log/%s.log" % (simdir, job_name))

        # import log file and fetch last dose
        logdata = np.loadtxt('%s/log/%s.log' % (simdir, job_name))
        dpadose = logdata[-1,1]
        iteration = len(logdata) - 1
    else:
        # otherwise, read initial structure file, if given
        if initial:
            announce("Starting from file: %s" % initial)
            if initialtype == "dump":
                initialframe = get_dump_frame(initial)
                if is_triclinic(initial):
                    lmp.command('run 0')
                    lmp.command('change_box all triclinic')
                lmp.command('read_dump %s %d x y z purge yes add yes box yes replace no' % (initial, initialframe))

            if initialtype == "data":
                lmp.command('read_data %s' % initial)
                
            lmp.command('reset_timestep 0')
        dpadose = 0.0
        iteration = 0

    # load potential
    if alloy:
        for _i,_m in enumerate(potential.mass.values()):
            lmp.command('mass %d %f' % (_i+1, _m)) 
        lmp.command('pair_style eam/alloy')
        lmp.command(('pair_coeff * * %s ' % potfile) + '%s '*nelements % tuple(potential.elements))
    else:
        lmp.command('mass            1 %f' % list(potential.mass.values())[atype])
        lmp.command('pair_style eam/fs')
        lmp.command('pair_coeff * * %s %s' % (potfile, list(potential.elements)[atype]))
    lmp.command('neighbor 1.5 bin')

    # initialise
    lmp.command('run 0')

    # time step in femtoseconds
    timestep = 0.002 
    lmp.command('timestep %f' % timestep)

    # update thermo and wrap atoms back into the box
    lmp.command('thermo_style custom step time temp press pe ke pxx pyy pzz pxy pxz pyz lx ly lz')
    lmp.command("thermo_modify line one format line '%8d %10.6f %10.6f %10.6f %15.8e %15.8e %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %6.2f %6.2f %6.2f'")

    lmp.command('thermo 50')

    # relax structure and box dimensions if no restartfile found 
    if not restartfile:

        # relax atomic coordinates only
        lmp.command('minimize %s 0 10000 10000' % (etolstring))

        # relax box dimensions too 
        if len(freebox) > 0:
            lmap = {"x": 0, "y": 1, "z": 2, "xy": 3, "yz": 4, "xz": 5}
            if len(freebox) == 6:
                lmp.command('fix ftri all box/relax tri 0.0 vmax 0.0001 nreset 100')
            else:
                for _vx in freebox:
                    lmp.command('fix f%sfree all box/relax %s %f vmax 0.0001 nreset 100' % (_vx, _vx, boxstress[lmap[_vx]]))

            lmp.command('min_modify line quadratic')
            lmp.command('minimize %s 0 10000 10000' % (etolstring))

            # fix box dimensions again
            if len(freebox) == 6:
                lmp.command('unfix ftri')
            else:
                for _vx in freebox:
                    lmp.command('unfix f%sfree' % _vx)


    lmp.command('run 0')

    if not restartfile:
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
        lmp.command("print '%d %f ${vpe} ${vpxx} ${vpyy} ${vpzz} ${vpxy} ${vpxz} ${vpyz} ${vlx} ${vly} ${vlz}' append %s/log/%s.log" % (iteration, 0.0, simdir, job_name))

    # print out first dump
    if all_input['write_data'] and not restartfile:
        lmp.command('write_dump all custom %s/%s/%s.%d.dump id type x y z' % (scrdir, job_name, job_name, iteration))
    
    # initialise NVE ensemble 
    lmp.command('fix fnve all nve')
    
    # initialise stopping model
    lmp.command('fix fstopping all electron/stopping 1e-9 %s' % stoppingfile)
    lmp.command('velocity all create 0.0 1 mom yes rot no')
    
    comm.barrier()
 
    # convert mass from Dalton to eV ps^2/Ang^2
    kickmass = mass * 1.03499e-4

    # define compute for max atomic movement in a time-step
    lmp.command('variable vvnorm atom sqrt(vx*vx+vy*vy+vz*vz)')
    lmp.command('compute cvmax all reduce max v_vvnorm')
    lmp.command('variable dmax equal c_cvmax*dt')

    lmp.command('thermo_style custom step dt time v_dmax temp press pe ke pxx pyy pzz pxy pxz pyz lx ly lz')
    lmp.command("thermo_modify line one format line '%8d %6e %10.6f %10.6f %10.6f %10.6f %15.8e %15.8e %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %6.2f %6.2f %6.2f'")
    lmp.command('run 0')

    # run with small steps first to avoid losing atoms due to large velocities
    lmp.command('thermo 50')

    # keep box centre of mass from drifting
    lmp.command("fix frecenter all recenter INIT INIT INIT")
    
    # defining parameters for the electronic damage model
    # here: alloys not yet implemented! 
    A1, A2, Z1, Z2 = mass, mass, znum, znum 
    mpiprint ("Initialising damage energy estimation using parameters A1, A2, Z1, Z2:", A1, A2, Z1, Z2)
   
    # lindhard electronic stopping model for damage energy
    tdmodel  = Lindhard(A1, A2, Z1, Z2)

    # damage energy correction due to 10 eV cutoff in simulation
    dE = 10-tdmodel(10.)

    # estimate melting energy from 6*kB*Tmelt (overestimation as no latent heat of fusion given)
    kB = 0.0000861733 # in eV/K
    emelt_guess = 6.*kB*Tmelt # in eV

    # define exclusion radius method 
    def exclusion_radius(PKAenergy):
        nmelt = PKAenergy/emelt_guess   # app. number of molten atoms
        vmelt = nmelt * .5*alattice**3  # app. melt volume in Ang^3
        rmelt = np.power(3./(4.*np.pi)*vmelt, 1./3.) # app. melt radius in Ang
        rbuffer = 5.0 # additional buffer in Ang
        exc_rad = rmelt + rbuffer # total exclusion radius in Ang
        return exc_rad
    exc_radius = np.vectorize(exclusion_radius)
    
    # run consecutive cascades 
    for cloop in range(1, int(1e6)):
    
        if dpadose >= maxdpa:
            announce ("Finished simulation. Current dose is %8.3f dpa, with target dose given by %8.3f dpa." % (dpadose, maxdpa))
            break 
 
        # extract cell dimensions 
        N = lmp.extract_global("natoms", 0)
        xlo, xhi = lmp.extract_global("boxxlo", 2), lmp.extract_global("boxxhi", 2)
        ylo, yhi = lmp.extract_global("boxylo", 2), lmp.extract_global("boxyhi", 2)
        zlo, zhi = lmp.extract_global("boxzlo", 2), lmp.extract_global("boxzhi", 2)
        xy, yz, xz = lmp.extract_global("xy", 2), lmp.extract_global("yz", 2), lmp.extract_global("xz", 2)

        # Relevant documentation: https://docs.lammps.org/Howto_triclinic.html 
        xlb = xlo + min(0.0,xy,xz,xy+xz)
        xhb = xhi + max(0.0,xy,xz,xy+xz)
        ylb = ylo + min(0.0,yz)
        yhb = yhi + max(0.0,yz)
        zlb, zhb = zlo, zhi

        lims_lower = np.r_[xlb, ylb, zlb]  # bounding box origin
        lims_upper = np.r_[xhb, yhb, zhb]  # bounding box corner
        lims_width = lims_upper-lims_lower # bounding box width

        # Basis matrix for converting scaled -> cart coords
        c1 = np.r_[xhi-xlo, 0., 0.]
        c2 = np.r_[xy, yhi-ylo, 0.]
        c3 = np.r_[xz, yz, zhi-zlo]
        cmat =  np.c_[[c1,c2,c3]].T
        cmati = np.linalg.inv(cmat)

        _x = np.ctypeslib.as_array(lmp.gather_atoms("x", 1, 3)).reshape(N, 3)
        _x = _x - np.array([xlo,ylo,zlo])

        # convert to fractional coordinates (einsum is faster than mat-mul) 
        _xfrac = np.einsum("ij,kj->ki", cmati, _x)

        if (me == 0):
            # incremental applied dose (using damage energy)
            appdose = 0.0
            doselimit = incrementdpa 

            mpiprint ("Draw random cascade energies until the minimal dose increment is reached.")

            # upper limit: no more than 1.2 times dose limit
            nattempts = 0
            incrtol = 0.0 
            while (appdose >= (1.2+incrtol)*doselimit) or appdose == 0.0:
                appdose = 0.0
                cascade_pka = []

                # lower limit: higher than dose limit
                while (appdose <= (1.0-incrtol)*doselimit):
                    epka = 0.0
                    while (epka <= pkamin) or (epka > pkamax):
                        epka = np.random.choice(pkas)
                    cascade_pka += [epka]

                    # convert pka energies to damage energies
                    tdams = np.array([tdmodel(_pka)+dE for _pka in cascade_pka])

                    # compute frenkel pairs produced in NRT model and update dose increment
                    ndefects = np.array([nrtdamage(_td, edavg) for _td in tdams])
                    appdose = np.sum(ndefects/N)

                nattempts += 1
                if nattempts >= 1000:
                    print ()
                    print ("Could not draw damage energies within the range of [%3.2f*doseincrement, %3.2f*doseincrement] after %d attempts!" % (1.0-incrtol, 1.2+incrtol, nattempts))
                    incrtol += 0.1
                    print ("Increasing interval range.")
                    print ()

            print ("Dose increment:", appdose)


            # sort PKA energies in descending order to ensure we can fit in the largest cascades
            cascade_pka = np.flip(np.sort(cascade_pka))
            ncascades = len(cascade_pka)

            mpiprint ("Initialising %d cascades leading to a dose increment of %9.5f dpa with energies (eV):" % (ncascades, appdose))
            print ("damage energies:")
            print (cascade_pka)
            mpiprint ()

            cascade_pos = []
            mpiprint ("Drawing random non-overlapping cascade positions.")

            count = 0
            while (len(cascade_pos) < ncascades):

                # first, create a trial cascade inside the bounding box
                trisuccess = False
                for nattempts in range(int(1e6)):
                    _trial_pos = lims_width*np.random.rand(3)

                    # next, check if the point lies inside the triclinic box, otherwise repeat
                    _trial_pos
                    _frac = cmati@_trial_pos
                    if (_frac > 0.0).all() and (_frac < 1.0).all():
                        trisuccess = True
                        break
                
                if not trisuccess:
                    annouce ("Error: could not place a random point inside triclinic cell after %d attempts." % nattempts)
                    return 1

                _ncasc = len(cascade_pos)
                if _ncasc == 0: 
                    # always accept the first cascade
                    cascade_pos += [_trial_pos]
                else:
                    # subsequent cascades need to be checked for overlap

                    # compute distance between trial cascade and all other cascades using minimum img convention
                    dr = cascade_pos - _trial_pos
                    df = np.array([cmati@dri for dri in dr])
                    df[df  > .5] -= 1.0
                    df[df <=-.5] += 1.0
                    dr = np.array([cmat@dfi for dfi in df]) # convert back to cartesian coordinates
                    dnorm = np.linalg.norm(dr, axis=1)

                    # get exclusion distances
                    excdist = exc_radius(cascade_pka[:_ncasc]) + exc_radius(cascade_pka[_ncasc])

                    # only accept cascade if all of the distances exceeed the exclusion distances
                    if (dnorm > excdist).all():
                        cascade_pos += [_trial_pos]

                if count >= 1e6:
                    mpiprint("Error: could not reach target dose after %d iterations (current dose: %8.5f)" % (count, appdose))
                    return 0
                count += 1

            cascade_pos = np.r_[cascade_pos]

            mpiprint ()
            mpiprint ("Selected cascade positions (Ang) and recoil energies (eV):")
            for _cp in range(ncascades):
                mpiprint ("(%8.3f, %8.3f, %8.3f)  %8.5f" % (cascade_pos[_cp][0], cascade_pos[_cp][1], cascade_pos[_cp][2], cascade_pka[_cp]))
            mpiprint ()

            # build KDTree (in fractional coords) for nearest neighbour search containing all atomic data
            xf_ktree = cKDTree(_xfrac, boxsize=[1,1,1])

            # find atoms nearest to cascade centres to apply kicks to
            kick_indices = [1+xf_ktree.query(cmati@_cpos, k=1)[1] for _cpos in cascade_pos]

            # kick velocities in Ang/ps (same as velocity in LAMMPS metal units)
            kickvelocities = np.sqrt(2.*cascade_pka/kickmass)
            _sph = sample_spherical(ncascades)
            kick_velocities = [kickvelocities[_i]*_sph[_i] for _i in range(ncascades)]
        else:
            cascade_pka = None
            kick_indices = None
            kick_velocities = None 
            appdose = None
        
        comm.barrier()
        cascade_pka = comm.bcast(cascade_pka, root=0)
        kick_indices = comm.bcast(kick_indices, root=0)
        kick_velocities = comm.bcast(kick_velocities, root=0) 
        appdose = comm.bcast(appdose, root=0) 
        ncascades = len(cascade_pka)

        dpadose += appdose

        # apply random kicks
        for _c in range(ncascades):
            _ki = kick_indices[_c]
            mpiprint("Atom ID %d at (%8.3f, %8.3f, %8.3f) gets %12.6f eV recoil energy (%12.6f A/ps)." % (_ki,
                    _x[_ki-1][0]+xlo, _x[_ki-1][1]+ylo, _x[_ki-1][2]+zlo, cascade_pka[_c], np.linalg.norm(kick_velocities[_c])))
        
            lmp.command('group gkick id %d' % kick_indices[_c])
            lmp.command('velocity gkick set %f %f %f sum yes units box' % tuple(kick_velocities[_c]))
            lmp.command('group gkick delete') 
        mpiprint ()

        # initially strict adaptive time-step
        lmp.command('fix ftimestep all dt/reset 1 NULL 0.002 0.01 units box')
        lmp.command('run 500 post no')

        # less strict later on
        lmp.command('unfix ftimestep')
        lmp.command('fix ftimestep all dt/reset 10 NULL 0.002 0.1 units box')
            
        maxloops = int(1e6)
        nsteps = 250
        announce("Running loops of %d steps while monitoring temperature." % nsteps)
           
        # simulation time value is not reset, so need to keep track of the difference 
        initial_time = float(lmp.get_thermo('time'))

        for k in range(maxloops):
            lmp.command('run %d post no' % nsteps) 
            current_temperature = float(lmp.get_thermo('temp')) 
            current_time = float(lmp.get_thermo('time'))

            if current_temperature <= tmax:
                if (current_time-initial_time) > minruntime:
                    mpiprint("Reached maximal temperature and minimum run time, propagation complete.")
                    break
                else:
                    mpiprint("Reached maximal temperature but not minimum run time, propagating more.")

        # finish off with CG minimisation and write dump file
        if runCG:
            lmp.command('velocity all create 0.0 1 mom yes rot no')
 
            # relax atomic coordinates only
            lmp.command('minimize %s 0 10000 10000' % (etolstring))

            # relax box dimensions too 
            if len(freebox) > 0:
                lmap = {"x": 0, "y": 1, "z": 2, "xy": 3, "yz": 4, "xz": 5}
                if len(freebox) == 6:
                    lmp.command('fix ftri all box/relax tri 0.0 vmax 0.0001 nreset 100')
                else:
                    for _vx in freebox:
                        lmp.command('fix f%sfree all box/relax %s %f vmax 0.0001 nreset 100' % (_vx, _vx, boxstress[lmap[_vx]]))

                lmp.command('min_modify line quadratic')
                lmp.command('minimize %s 0 10000 10000' % (etolstring))

                # fix box dimensions again
                if len(freebox) == 6:
                    lmp.command('unfix ftri')
                else:
                    for _vx in freebox:
                        lmp.command('unfix f%sfree' % _vx)

            # this is needed to wrap atoms back into the box
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

        lmp.command('unfix ftimestep')
        lmp.command('reset_timestep 0')

        # print recoil energy log file
        pka_string = "print '%d " % (iteration+cloop) + "%10.6f "*len(cascade_pka) % tuple(cascade_pka) +  "' append %s/log/%s.pka" % (simdir, job_name)
        lmp.command(pka_string)

        # write restart file always
        dfile = "%s/%s/%s.restart" % (scrdir, job_name, job_name)
        announce("Writing restart file %s." % dfile)
        comm.barrier()
        lmp.command('write_dump all custom %s id type x y z' % dfile) 
        comm.barrier()

        if ((iteration + cloop) % export_nth) == 0:
            dfile = "%s/%s/%s.%d.dump" % (scrdir, job_name, job_name, iteration+cloop)
            announce("Writing dump file %s." % dfile)
            comm.barrier()
            lmp.command('write_dump all custom %s id type x y z' % dfile)     

        comm.barrier()

    lmp.close()
    return 0


if __name__ == "__main__":
    main()

    if mode == 'MPI':
        MPI.Finalize()
