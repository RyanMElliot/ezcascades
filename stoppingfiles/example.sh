#!/bin/sh
mpirun -n 1 python make_stopping_table.py 3 Ni-FeNiCr.txt Fe-FeNiCr.txt Cr-FeNiCr.txt Ni Fe Cr NiFeCr NiFeCr.dat
mpirun -n 1 python make_stopping_table.py 1 Fe-FeNiCr.txt Fe Fe-solo Fe-solo.dat
