import numpy as np
from geodesicDome.geodesic_dome import GeodesicDome

# increase this if you need to generate geodesic coordinates with resolution
MAXORDER = 6

def main():

    # construct a geodesic dome for sampling E_d on the unit sphere
    for freq in range(MAXORDER):
        gdome = GeodesicDome(freq=freq)
        kvec = gdome.get_vertices()
        np.savetxt("geo.%d.dat" % freq, kvec)

    return 0

if __name__=="__main__":
    main ()
