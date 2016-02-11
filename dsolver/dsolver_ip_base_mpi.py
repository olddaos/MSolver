#!/usr/bin/env python
"""
Base Distributed Solver to train Convolutional Neural Networks based on Caffe library using IPython and PyCUDA GPUArray
"""
import importlib
import numpy as np
from dsolver_ip_base import IPSolverBase
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
#from lockfile import LockFile
from mpi4py import MPI
import time
import logging
import os

class IPSolverBaseMPI(IPSolverBase):
    """
    Base class for all IPython based distributed solvers.
    It is responsible for initialization, controls active engines, etc.
    This is abstract class. Solve method is not implemented.
    """

    def __init__(self, local_solver_cls, dsolver_config_file):
        """
        Initializes remote clients for ipython engines.

        **Args**:
           cluster_config_file (str) : path to cluster configuration file, which
                                       includes information about models prototypes,
                                       ipython cluster configuration, datasets location, and so on.
        """

        IPSolverBase.__init__(self, local_solver_cls, dsolver_config_file)

    def summary(self):
        """
        Returns summary about current DSolver session.
        """

        info = {}
        info['DSolver Communication Layer'] = 'IPython.parallel'
        info['DSolver engine'] = self.local_solver_cls.__name__
        if self._cluster_config.has_key('model_def_file'):
            name = os.path.basename(self._cluster_config['model_def_file'])
        else:
            name = os.path.basename(self._cluster_config['solver_def_file'])
        info['Model Name'] = name
        info['Train DB'] = os.path.basename(self._cluster_config['train_db_name'])
        info['Number of engines'] = str(len(self.active_engines))

        current_iter = self.rc[self.master_id]['local_solver.iter']
        max_iter = self.rc[self.master_id]['local_solver.max_iter']
        info['Progress'] = str(100 * current_iter/ max_iter)+'%'
        info['Total Iterations'] = str(max_iter)
        info['Current Iteration'] = str(current_iter)
        if hasattr(self, '_solve_start'):
            info['Time Elapsed'] = "%.2f hours" % ((time.time() - self._solve_start) / 3600)
        info['Train Loss'] = "%.4f" % self.rc[self.master_id]['local_solver.train_loss']
        #info['Test Score'] = "%.4f" % (self.log['val']['score'][-1][0][0] if self.log['val']['score'] else 0)
        info['DSolver engines IDs'] = str(self.active_engines)
        return json.dumps(info)

    def _import_pycuda_util(self, config):
        # Import additional pycuda_util
        self.facade_pycuda_util = importlib.import_module('caffe_facade.pycuda_util')
        #facade_pycuda_util = importlib.import_module(cluster_config['facade_name'] + ".pycuda_util")

    def _init_pycuda_arrays(self):
        """
        Initializes repote engines. Imports, initializes globals, sets engines ids.
        """
        import atexit

        # Init context
        self.ctx = self.facade_pycuda_util.CaffeCudaContext()

        # Init Cublas Handle
        self.cublas_handle = self.facade.cublas_handle()
        import cffi
        ffi = cffi.FFI()
        self.to_buffer = lambda arr: None if arr is None else ffi.buffer(ffi.cast("void*", arr.ptr), arr.nbytes)
        # FIXME:  don't know why we ever need that. mpi.Bcast( a_gpu.get(), rank=0 ) works good

        # Init blobs
        self.blobs = list()
        for (blob_name, blob)  in self.local_solver.net.params.items(): self.blobs.append(blob[0]); self.blobs.append(blob[1])

        self.data_blobs_gpu_initial = list()
        for blob in self.blobs: self.data_blobs_gpu_initial.append(gpuarray.to_gpu(np.zeros_like(blob.data).astype(np.float32)))

        self.temp_buffer_tosync = list()
        for blob in self.blobs: self.temp_buffer_tosync.append(gpuarray.to_gpu(np.zeros_like(blob.data).astype(np.float32)))

        # Init necessary GPUArrays
        self.data_blobs_gpu = [blob.data_as_pycuda_gpuarray() for blob in self.blobs]
        self.diff_blobs_gpu = [blob.diff_as_pycuda_gpuarray() for blob in self.blobs]

