import numpy as np

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

        fp.readline()
        fp.readline()
        fp.readline()

        # also, check next line and see if atom types are included
        string = fp.readline()

        flag = None
        if " type " not in string:
            flag = "Warning: no atomic type column found in %s. Command read_dump will fail!" % fpath
    return tri, flag


