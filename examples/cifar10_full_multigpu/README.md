# Step by step guide how to run CIFAR10 Distributed Training Demo

## Overview
This demo demonstrates the main functionality of distributed training on 8 GPUs solver for Convolutional Neural Network model on CIFAR10 dataset. It will run locally on your machine with 8 worker engines.

## Prerequisites

To properly run this demo you will need *caffe-rc2* from [Caffe webpage](https://github.com/BVLC/caffe/releases). Let's suppose that it locates at **CAFFE-RC2_DIR**.
Take a note that this version of *caffe* is working only with *CUDNN v1*.

Also you will need installed *IPython* with *PyZMQ*.

One of the main part to run distributed training is *dsolver*. Let's assume that it locates at **DSOLVER_DIR**.

## Steps to run demo

1. Build caffe library:
```shell
cd $CAFFE-RC2_DIR
```
At this step you should edit **Makefile.config** to specify dependencies directories and other build options.
```shell
make -j8
make py
```
2. You need CIFAR10 dataset to run demo. To download dataset you can follow caffe instructions:
To get CIFAR10 dataset just run the script in shell:
```shell
sh ./data/cifar10/get_cifar10.sh
```
It could take 5 minutes or so to download the dataset.
Then to convert CIFAR10 data to LMDB format:
```shell
sh examples/cifar10/create_cifar10.sh
```
3. Now it is time to build *facade* for *dsolver*:
```shell
cd $DSOLVER_DIR/facades/caffe_facade
```
Here you might need to check **Makefile.config** that all depedencies paths are correct. Make sure that
*CAFFE_DIR* points to the right place. In general you should use almost the same setting as in **Makefile.config** for *caffe*.
```
make
```
4. Prepare *Ipython* profile:
```shell
cd $DSOLVER_DIR/examples/cifar10_full_multigpu/
sh ./create_profile.sh
```
This is not mandatory in general case but made to not clutter your main IPython directory.

5. Make sure that paths in **cifar10_full_multigpu_local.json** are correct and point to CIFAR10 dataset files.

6. You have 2 options how to run training: in **synchronous** mode and in **asynchronous** mode.
Use **config.py** file to configure *dsolver*. To run *dsolver* in **synchronous** mode set:

```python
...
DSOLVER_CLASS = 'SDSolver'
...
```
To run *dsolver* in **asynchronous** mode set:
```python
...
DSOLVER_CLASS = 'ADSolver'
...
```

To run distributed training:
```shell
sh start_training.sh
```
This script will run IPython cluster with 8 engines and will initiate training on CIFAR10 dataset.
Application automatically creates web-page with training session summary, which is available at http://localhost:8080
Also you can check the log files **log\_cifar10\_*** (in [Pickle](http://docs.python.org/2/library/pickle.html) format).
Test score and Train Loss updated according to the settings in **cifar10_full_solver.prototxt**.

## Configuration

1. By default demo runs on 8 GPUs with IDs from 1 to 8. GPU with ID 0 is used as master (parameter server).
2. If you would like to change GPU IDs or/and number of GPUs, please do the following steps:
  * edit field **gpu_ids** in **cifar10_full_multigpu_local.json**:
```json
...
    "gpu_ids": [1, 2, 3, 4, 5, 6, 7, 8]
...
```
  * make sure that IPCluster is initialized with sufficient number of engines (parameter **n** should be higher or equal to number of GPUs in **start_training.sh**).
