#! /bin/bash

#source dsolver.conf
mpiexec -n 3 ./test_mpi.py -u > mpi_log.log

echo "After MPI run"
sleep 30


