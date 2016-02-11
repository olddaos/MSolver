#!/usr/bin/env python
"""
Asynchronous and Synchronous Distributed Solvers for training Convolutional Neural Networks based on Caffe library using IPython.
"""

import time
import datetime
from   dsolver_ip_base_mpi import IPSolverBaseMPI
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import scikits.cuda.cublas as cublas
import numpy as np
from mpi4py import MPI
import os
#from memory_profiler import profile
#from guppy import hpy
import gc
#hp = hpy()
#import objgraph

#hp.setrelheap()


#def memory_usage_psutil(  ):
           # return the memory usage in MB
#           import psutil
 #          process = psutil.Process(os.getpid())
  #         mem = process.get_memory_info()[0] / float(2 ** 20)
   #        return mem

class SDSolverMPI(IPSolverBaseMPI):
        """
        Synchronous Distributed Solver.
        """
        def __init__(self, local_solver_cls, dsolver_config_file):
           IPSolverBaseMPI.__init__(self, local_solver_cls, dsolver_config_file)

	def solve(self):
	   if self.rank == 0:	
               self.logger.info('SDSolverMPI started...')
	       self.logger.info('Current Datetime = {0}'.format(str(datetime.datetime.now())))
	   self._solve_start = time.time()

           print "Doing initial output..."
	   # FIXME FIXME FIXME
	   iter = 0
	   max_iter = 17500 #60000 #450000

	   while iter < max_iter:
              print 'Iter {0:d} from {1:d}...'.format(iter, max_iter)
	      if self.rank == 0:
		  print 'Iter {0:d} from {1:d}...'.format(iter, max_iter)
	          self.logger.info('Iter {0:d} from {1:d}...'.format(iter, max_iter))

	      self.compute_weights_updates( iter)
	      # Update train loss
	      self.reduce_obtained_updates()

	      #One weights update - one iteration
	      iter += 1

              _ = gc.collect()

	      # Logging and snapshots
	      if self.rank == 0:
		print self.local_solver.train_loss
	      	self._log_training()

	   # Make a snapshot on master
	   if self.rank == 0:
               print "Snapshotting..."
               for (data, data_blob_gpu) in zip( self.local_solver.net.params_data, self.data_blobs_gpu): 
	           cuda.memcpy_dtoh(data, data_blob_gpu.ptr)
	           self.local_solver.snapshot()
	           self.local_solver.output_finish()

	def compute_weights_updates( self, current_iteration ):
	    """
	    Receives whole network weights in GPU memory using IPC.
	    Assigns received weights to network of local_solver.
	    Then runs for gradient_sync_interval iterations ForwardBackward operation for updating local weights.
	    Locally updated weights is the result of the function.

	    **Args**:
		current_iteration (int) : current iteration.
	    """
	    self.ctx.synchronize()
	    self.local_solver.iter = current_iteration
	    for data_blob in self.data_blobs_gpu:
		if self.rank == 0:
		    for i in range(1,self.comm_size):
			self.comm.Send([ self.to_buffer(data_blob) ,MPI.FLOAT], dest=i)
		else:
		    self.comm.Recv([ self.to_buffer(data_blob) ,MPI.FLOAT], source=0)
		#self.comm.Bcast([ self.to_buffer(data_blob) ,MPI.FLOAT], root=0)
	    self.ctx.synchronize()
	    if self.rank > 0:
		# Run training network for num_iterations
		compute_time = time.time()
		for i in range(self._gradient_sync_interval):
			# TODO TODO FIXME FIXME: forward_backward spawns MEMORY LEAK ( either in Caffe and ( probably ) in facade . People relate this error to DataReaders, which supposed to be root cause 
			self.local_solver.forward_backward()
			self.local_solver.compute_update_value()
			self.local_solver.update()

		self.ctx.synchronize()

		self.local_solver.calculate_train_info();
		compute_time = time.time() - compute_time
	    else:
		self.local_solver.train_loss = 0.0

            print "Rank # %d before reduce" % self.rank
	    dummy = self.comm.reduce( self.local_solver.train_loss, op=MPI.SUM, root=0)
            print "Rank # %d after reduce" % self.rank

	    if self.rank == 1:
		self.comm.send( self.local_solver.train_info, dest=0)
	    elif self.rank == 0:
		self.local_solver.train_loss = dummy
		self.local_solver.train_loss /= self.comm_size-1
		self.local_solver.train_info = self.comm.recv(source=1)

	def reduce_obtained_updates( self ):
	    if self.rank == 0:
		for data_blob in self.data_blobs_gpu:
	# FIXME	    data_blob.fill(0)
		    cublas.cublasSscal( self.cublas_handle, data_blob.size, 0, data_blob.gpudata, 1)
		self.ctx.synchronize()
	   
	    for i in xrange(len( self.data_blobs_gpu)):
		if self.rank == 0:
		    for j in range(1,self.comm_size):
		        self.comm.Recv([ self.to_buffer( self.data_blobs_gpu_initial[i]), MPI.FLOAT], source=MPI.ANY_SOURCE)
		        cublas.cublasSaxpy(self.cublas_handle, self.data_blobs_gpu_initial[i].size, 1.0, self.data_blobs_gpu_initial[i].gpudata, 1, self.data_blobs_gpu[i].gpudata, 1) 
		    #self.comm.Reduce(MPI.IN_PLACE, [ self.to_buffer( self.data_blobs_gpu[i]), MPI.FLOAT], op=MPI.SUM, root=0)
		else:
		    self.comm.Send([ self.to_buffer( self.data_blobs_gpu[i]), MPI.FLOAT], dest=0)
		    #self.comm.Reduce([ self.to_buffer( self.data_blobs_gpu[i]), MPI.FLOAT], [ self.to_buffer( self.data_blobs_gpu[i]), MPI.FLOAT], op=MPI.SUM, root=0)
		self.comm.Barrier()
	    self.ctx.synchronize()
	    
	    if self.rank == 0:
		for data_blob in self.data_blobs_gpu:
		    cublas.cublasSscal(self.cublas_handle, data_blob.size, 1.0 / (self.comm_size-1), data_blob.gpudata, 1)
		self.ctx.synchronize()

	def initial_output( self ):
	    # Initial output
	    if self.rank == 0:
		self.local_solver.test()
		self.local_solver.output_train_loss()
		self.local_solver.calculate_train_info()
		self.local_solver.output_train_info()
		self.local_solver.output_learning_rate()

	def snapshot_sync_cpu( self ):
	    if self.local_solver.snapshot_interval and\
	    self.local_solver.iter - iter_old['snapshot'] >= self.local_solver.snapshot_interval:
		for (data, data_blob_gpu) in zip( self.local_solver.net.params_data, self.data_blobs_gpu):
		     cuda.memcpy_dtoh(data, data_blob_gpu.ptr)


