#! /bin/bash

export ORTED_PATH=<set up absolute path to your ORTED daemon>
TOTAL_NODES=32
PER_NODE=16

nohup mpirun --mca btl_base_verbose  100 -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH -x PATH=$PATH --mca btl_openib_want_cuda_gdr 1  --mca mpi_common_cuda_event_max 2500 --mca btl openib,smcuda,self -launch-agent $ORTED_PATH -n $TOTAL_NODES -N $PER_NODE -hostfile dsolver_machinefile python train.py > async_full.log 2> async_full.err
