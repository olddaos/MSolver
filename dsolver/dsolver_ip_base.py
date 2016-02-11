#!/usr/bin/env python
"""
Base Distributed Solver to train Convolutional Neural Networks based on Caffe library using IPython
"""

import sys
import os
sys.path.insert(0, '/home/al.miasnikov/.conda/envs/local_conda/lib/python2.7/site-packages/scikits')

import numpy as np
import time
import datetime
import json
import shutil
import socket
import importlib
from dsolver_base import DSolverBase


IFL = '....'

class IPSolverBase(DSolverBase):
    """
    Base class for all IPython based distributed solvers.
    It is responsible for initialization, controls active engines, etc.
    This is abstract class. Solve method is not implemented.
    """
    def __init__(self, local_solver_cls, dsolver_config_file):
        """
        Initializes remote clients for ipython engines.

        **Args**:
           cluster_config_file (str) : path to cluster configuration file, which includes information about models prototypes,
                                       ipython cluster configuration, datasets location, and so on.
        """

        DSolverBase.__init__(self, local_solver_cls, dsolver_config_file)
        self._cluster_config = self._config['cluster']
        self._training_config = self._config.get('training', dict())
        conn_file = self._cluster_config.get('cluster_connector_file', None)
        ipcluster_profile = self._cluster_config.get('ipcluster_profile', None)
        self._gradient_sync_interval = self._training_config.get('gradient_sync_interval', 1)
        self._warm_start = self._training_config.get('warm_start', False)
        self._warm_start_resume_file = self._cluster_config.get('warm_start_resume_file', None)
        if self._warm_start_resume_file:
            self._warm_start_resume_file = self._warm_start_resume_file.encode('ascii', 'ignore')

	# TODO: Compute rank here
        self._finished_tasks = set()

    def summary(self):
        """
        Returns summary about current DSolver session.
        """

        info = {}
        info['DSolver Communication Layer'] = 'IPython.parallel'
        info['DSolver engine'] = self.local_solver_cls.__name__
        if self._cluster_config.has_key('model_def_file'):
            name = self._cluster_config['model_def_file']
        else:
            name = self._cluster_config['solver_def_file']
        info['Model Name'] = name
        info['Train DB'] = self._cluster_config['train_db_name']
        info['Number of engines'] = str(len(self.active_engines))
        info['Progress'] = str(100*self.local_solver.iter / self.local_solver.max_iter)+'%'
        info['Total Iterations'] = str(self.local_solver.max_iter)
        info['Current Iteration'] = str(self.local_solver.iter)
        if hasattr(self, '_solve_start'):
            info['Time Elapsed'] = "%.2f hours" % ((time.time() - self._solve_start) / 3600)
        info['Train Loss'] = "%.4f" % self.local_solver.train_loss
        #info['Test Score'] = "%.4f" % (self.log['val']['score'][-1][0][0] if self.log['val']['score'] else 0)
        info['DSolver engines IDs'] = str(self.active_engines)
        return json.dumps(info)

    def deploy(self):
        """
        Deploy training on all nodes.
        Initializes working directories, creates symbolic links to dataset,
        copies intermediate files to working directory.
        """

        if 'master' in self._config:
            config = self._config['master']
        else:
            config = self._cluster_config


    def init_engines(self):
        """
        Initializes repote engines. Imports, initializes globals, sets engines ids.
        """
        import sys
        import os
        import socket
        import numpy as np
        import time
        import importlib
        # Import PyCUDA GPUArray and Cublas and MPI
        import pycuda.gpuarray as gpuarray
        import scikits.cuda.cublas as cublas
        import pycuda.driver as cuda
        import atexit
        from mpi4py import MPI
	atexit.register(MPI.Finalize)
	self.comm = MPI.COMM_WORLD
	self.comm_size = self.comm.Get_size()
	self.rank = self.comm.Get_rank()

    def assign_gpus(self):
        pass
	# TODO:
        # Assign different GPUs for different engines
        #if hasattr(self, 'gpu_ids'):
        #        gine].execute('facade.set_device({})'.format(self.gpu_ids[ rank % len(self.gpu_ids)]))
        #        self.logger.info('Engine {0} works on {1} GPU'.format(engine, self.gpu_ids[i % len(self.gpu_ids)]))

    def init_solver(self):
        """
        Initializes solvers on remote engines
        """

        if 'master' in self._config:
            config = self._config['master']
        else:
            config = self._cluster_config

        sys.path.insert(0, os.path.abspath(config['dependencies_path']))
        sys.path.insert(0, os.path.abspath(os.path.join(config['dependencies_path'], config['facade_name'])))
        #os.chdir(self._working_dir)
        self.facade = importlib.import_module(config['facade_name'])
	self.gpu_ids =  self._cluster_config.get('gpu_ids', [0])#[ 8, 9, 10, 11, 8, 9, 10, 11 ]

        print "Init_solver: GPU_IDS : " + str(self.gpu_ids)

	self.facade.set_device( self.gpu_ids[ self.rank ] )
	print "DL02: Setting device # " + str( self.gpu_ids[ self.rank ])

        self.local_solver = self.local_solver_cls(config['solver_def_file'].encode('ascii', 'ignore'))

        # Import pycuda_util from facade if needed
        self._import_pycuda_util(config)

   #     if (self._warm_start):
   #         if self.mode != 'multiGPU_MPI':
   #             self.local_solver.init_solve(self._warm_start_resume_file)
   #     else:
  #          if self.mode != 'multiGPU_MPI':
        # FIXME : maybe its unnecessary
        self.local_solver.init_solve('')

       # Init pycuda necessary arrays if needed
        self._init_pycuda_arrays()

        self._training_log_reset()

    def _import_pycuda_util(self, config):
        # Not implemented in plain version
        pass

    def _init_pycuda_arrays(self):
        # Not implemented in plain version
        pass

    def clean(self):
        """
        Clean all contents of working directory and engines namespace.
        """
        #if os.access(self._cluster_config['working_dir'], os.F_OK):
        #    shutil.rmtree(self._cluster_config['working_dir'])
        #os.mkdir(self._cluster_config['working_dir'])

    def solve(self):
        """
        For base class this function is not implemented.
        """
        raise "Not implemented"

