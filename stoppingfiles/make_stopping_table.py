import numpy as np
import sys, os

def get_stoppingtable (path):
    assert os.path.isfile(path), "Error: file %s not found." % path

    with open(path, 'r') as fopen:
        rawdata = []

        ll = ''
        right_units = False
        while "-----------" not in ll:
            ll = fopen.readline()
            if "Stopping Units =   eV / Angstrom" in ll:
                right_units = True
        
        assert right_units, "Error: it appears the stopping power is not in units of eV / Angstrom!"

        ll = fopen.readline() # read one more to skip ---- line

        while "-----------" not in ll:
            rawdata += [ll]    
            ll = fopen.readline()

    processing = np.array([row.strip().split()[:3] for row in rawdata])
    multiplier = [float(row.replace('MeV', '1e6').replace('keV', '1e3').replace('eV', '1')) 
                  for row in processing[:,1]]

    stoppingtable = np.c_[multiplier * processing[:,0].astype(float), processing[:,2].astype(float)]
    
    return stoppingtable

def merge_stoppingpowers (mergelist, mergelabels, mergecompound, mergepath):

    mergelist = np.array(mergelist)

    eq_check = True
    for i in range(len(mergelist)-1):
        eq_check *= (mergelist[i][:,0] ==  mergelist[i+1][:,0]).all()

    assert eq_check, "Error: it appears the stopping tables are defined over differing ion energies."

    merged_data = np.vstack([mergelist[0,:,0], mergelist[:,:,1]]).T

    with open(mergepath, 'w') as fopen:
        fopen.write ('# Electronic stopping power in target %s for LAMMPS fix electron/stopping\n' % mergecompound)
        fopen.write ('# %8s   ' % '  ' + ' %18s'*len(mergelist) % tuple(mergelabels) + '\n')
        fopen.write ('# %10s  ' % 'eV' + ' %18s'*len(mergelist) % tuple(['eV/Ang']*len(mergelist)) + '\n')

        for row in merged_data:
            fopen.write ("%18.8f "*len(row) % tuple(row) + '\n') 


def main ():

    ntables = int(sys.argv[1])

    stopping_paths = sys.argv[2:2+ntables]
    ion_labels = sys.argv[2+ntables:2+2*ntables]
    compound_label = sys.argv[2+2*ntables]
    export_path = sys.argv[3+2*ntables]

    print ("Trying to import and merge SRIM stopping files:", stopping_paths)
    print ("The files are, respectively, for the ions:", ion_labels)
    print ("The resulting merged electroinc stopping table is for compound:", compound_label)
    print ("The LAMMPS stopping file will be exported to file:", export_path)
    print ()

    merge_tables = [get_stoppingtable (_path) for _path in stopping_paths] 
    merge_stoppingpowers (merge_tables, ion_labels, compound_label, export_path)



if __name__ == "__main__":
    main ()

