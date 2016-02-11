===============================
MultiGPU Convolutional Networks Training Framework
===============================

Uses OpenMPI to run distributed training of convolutional networks using Caffe as a backend.
Supports training on multiple hosts with multiple GPUs installed in them and uses OpenMPI support of Infiniband to speed up the computation ( when ran on HPC clusters ).
Tested on Amazon EC2 machines as well as on custom HPC cluster.



Usage
-----

First, you need to set up Caffe as described in their manual
Second, you need to create LMDB for your dataset using respective Caffe tools.
Then put the path to this LMDB ( as well as other paths ) in imagenet1000_sync.json
Also fix config.py to point respective pathes to where you need them

Input network description is in imagenet1000_sync.json

**Example Usage**
    ``$./train.sh``


Requirements
------------
* Caffe
* numpy
* scipy
* Docker

(may have to be independently installed) 


Installation
------------

```
  Check out stuff from GIT 
  Run Docker file to obtain a Docker VM, ready for use 

```
