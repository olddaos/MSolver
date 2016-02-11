#!/usr/bin/env python
"""
Asynchronous and Synchronous Distributed Solvers for training Convolutional Neural Networks based on Caffe library using IPython.
"""

import time
import datetime
from   dsolver_mpi_base import MPISolverBase
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import scikits.cuda.cublas as cublas
import numpy as np
from mpi4py import MPI
import os
import gc
import math
from  os.path import isfile

class MM_SDSolverMPI(MPISolverBase):
        """
        Synchronous Distributed Solver.
        """
        def __init__(self, local_solver_cls, dsolver_config_file):
           MPISolverBase.__init__(self, local_solver_cls, dsolver_config_file)
	   if self._cluster_config.has_key('number_nodes'):
               self.num_masters = self._cluster_config['number_nodes']

	def solve(self):
	   #This must be replaced in future by rank with highest possible IB speed
	   self.is_local_master = self.splitted_rank == 0
	   self.is_global_master = self.is_local_master & (self.rank == 0)
	   if self.is_local_master:
		print "I am master %d with padavans %d" % (self.rank, len(self.gpu_ids))
	   self.splitted_size = self.comm_splitted.Get_size()
           self.chunk_size = len(self.gpu_ids)
	   self.comm_masters = self.comm.Split(color=self.splitted_rank == 0, key=self.rank)
	   if self.is_local_master:
	       self.other_master_ranks = [r for r in  range(self.num_masters) if r != self.comm_masters.Get_rank()]

	   if self.is_global_master:
               self.logger.info("MM_SDSolverMPI started at submaster #%d..." % self.rank)
	       self.logger.info('Current Datetime = {0}'.format(str(datetime.datetime.now())))
	   self._solve_start = time.time()

	   iter = self.local_solver.iter 
	   max_iter = self.local_solver.max_iter
	   if self.is_local_master:
	       for i in xrange(len( self.data_blobs_gpu)):
	           for other_rank in self.other_master_ranks:
                       self.comm_masters.Sendrecv( [ self.to_buffer( self.data_blobs_gpu[i]), MPI.FLOAT], dest=other_rank, recvbuf=[ self.to_buffer( self.temp_buffer_tosync[i]), MPI.FLOAT], source=other_rank )
                       cublas.cublasSaxpy(self.cublas_handle, self.temp_buffer_tosync[i].size, 1.0, self.temp_buffer_tosync[i].gpudata, 1, self.data_blobs_gpu[i].gpudata, 1)
                   cublas.cublasSscal(self.cublas_handle, self.data_blobs_gpu[i].size, 1.0 / (self.num_masters), self.data_blobs_gpu[i].gpudata, 1)
               self.ctx.synchronize()
           self.comm.Barrier()
	   
	   while iter < max_iter:
	      if self.is_global_master:
		  print 'Iter {0:d} from {1:d}...'.format(iter, max_iter)
	          self.logger.info('Iter {0:d} from {1:d}...'.format(iter, max_iter))

	      self.compute_weights_updates( iter )
	      
	      self.reduce_obtained_updates(iter)
	      _ = gc.collect()

	      iter += 1
	      if self.is_global_master & (iter % self.local_solver.snapshot_interval == 0):
                  print "Snapshotting..."
                  for (data, data_blob_gpu) in zip( self.local_solver.net.params_data, self.data_blobs_gpu):
                      cuda.memcpy_dtoh(data, data_blob_gpu.ptr)
                      #self.local_solver.snapshot()
                      #self.local_solver.output_finish()
	      # Logging and snapshots
              if self.is_global_master &  (iter % self._master_sync_interval == 0):
                  print 'Loss: ' + str(self.local_solver.train_loss)
                  self._log_training()
	      if isfile("KILL_DSOLVER"):
	          #os.remove("KILL_DSOLVER")
		  break
		  

	   # Make a snapshot on master
	   if self.is_global_master:
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
	    self.local_solver.iter = current_iteration
	    for data_blob in self.data_blobs_gpu:
		self.bcast_log_local(0, data_blob)
	   
		#if self.is_local_master:
		#    requests = [MPI.REQUEST_NULL] * self.chunk_size	
		#    for i in range(1, self.chunk_size ):   # This ugly thing simply means, that we send blob to the slave nodes, residing at same root complex, where current submaster resides
			#requests[i] = self.comm_splitted.Isend([ self.to_buffer(data_blob) ,MPI.FLOAT], dest=i)
		    #MPI.Request.Waitall(requests)
		#else:
		#    slave_req = self.comm_splitted.Irecv([ self.to_buffer(data_blob) ,MPI.FLOAT], source=0)
                #    re=MPI.Request.Wait( slave_req )
		#self.comm_splitted.Barrier()
	   
	    self.ctx.synchronize()
	    compute_time = time.time()
	    for i in range(self._gradient_sync_interval):
	        self.local_solver.forward_backward()
	        self.local_solver.compute_update_value()
	        self.local_solver.update()

	    self.ctx.synchronize()

	    self.local_solver.calculate_train_info();
	    compute_time = time.time() - compute_time

	    if not self.is_local_master:
		to_send = np.array([self.local_solver.train_loss])
		self.comm_splitted.Send([to_send, MPI.FLOAT], dest=0,tag=self.comm_splitted.Get_rank())
	    else:
		dummy = np.array([0.])
		for i in range(1, self.chunk_size):
		    self.comm_splitted.Recv([dummy, MPI.FLOAT], source=MPI.ANY_SOURCE, tag=i)
		    self.local_solver.train_loss += dummy[0]
	    if self.is_local_master:
		self.local_solver.train_loss /= self.chunk_size

	def reduce_obtained_updates( self, iter ):
	   
	    for (data_blob, data_blob_temp) in zip(self.data_blobs_gpu, self.data_blobs_gpu_initial):
		self.reduce_log_local(0, data_blob, data_blob_temp) 
		#if self.is_local_master:
		#    for j in range( 1, self.chunk_size ):
		#        self.comm_splitted.Recv([ self.to_buffer( data_blob_temp), MPI.FLOAT], source=MPI.ANY_SOURCE)
		#        cublas.cublasSaxpy(self.cublas_handle, data_blob_temp.size, 1.0, data_blob_temp.gpudata, 1, data_blob.gpudata, 1) 
		#else:
		#    self.comm_splitted.Send([ self.to_buffer( data_blob), MPI.FLOAT], dest=0)

		#self.comm_splitted.Barrier()
	    self.ctx.synchronize()
	    
	    if self.is_local_master:
		for data_blob in self.data_blobs_gpu:
		    cublas.cublasSscal(self.cublas_handle, data_blob.size, 1.0 / self.chunk_size, data_blob.gpudata, 1)
		self.ctx.synchronize()
	    
	    if  self.is_local_master & (iter % self._master_sync_interval == 0):
		for i in xrange(len( self.data_blobs_gpu)):
                    for other_rank in self.other_master_ranks:
                        self.comm_masters.Sendrecv( [ self.to_buffer( self.data_blobs_gpu[i]), MPI.FLOAT], dest=other_rank, recvbuf=[ self.to_buffer( self.temp_buffer_tosync[i]), MPI.FLOAT], source=other_rank )
                        cublas.cublasSaxpy(self.cublas_handle, self.temp_buffer_tosync[i].size, 1.0, self.temp_buffer_tosync[i].gpudata, 1, self.data_blobs_gpu[i].gpudata, 1)
                    cublas.cublasSscal(self.cublas_handle, self.data_blobs_gpu[i].size, 1.0 / self.num_masters, self.data_blobs_gpu[i].gpudata, 1)
		loss = np.array([0.])
	        for other_rank in self.other_master_ranks:
		    temp = np.array([0.])
		    to_send = np.array([self.local_solver.train_loss])
		    self.comm_masters.Sendrecv([to_send, MPI.FLOAT], dest=other_rank, recvbuf=[temp, MPI.FLOAT], source=other_rank)
		    loss[0] += temp[0]
	        self.local_solver.train_loss = (self.local_solver.train_loss + loss[0])/self.num_masters
	    #self.comm.Barrier()
	
	def bcast_log_local(self, root, blob):
	    send_ranks = [root]
	    receive_ranks = [int(math.ceil(self.splitted_size/2))]
	    num_iter = int(math.ceil(math.log(self.splitted_size,2)))
	    for i in range(num_iter):
		for (s_rank, r_rank) in zip(send_ranks, receive_ranks):
		    if self.splitted_rank == s_rank:
		        master_req = self.comm_splitted.Isend([ self.to_buffer(blob) ,MPI.FLOAT], dest=r_rank)
			re=MPI.Request.Wait( master_req )
		    elif self.splitted_rank == r_rank:
			slave_req = self.comm_splitted.Irecv([ self.to_buffer(blob) ,MPI.FLOAT], source=s_rank)
                    	re=MPI.Request.Wait( slave_req )
		send_ranks.extend(receive_ranks) 
		receive_ranks = []
                for s_rank in send_ranks:
                    if s_rank + (2 ** i) < self.splitted_size:
                        receive_ranks.append(s_rank + (2 ** i))
		    else:
  		        send_ranks.remove(s_rank)
		self.comm_splitted.Barrier()	       

	def reduce_log_local(self, root, blob, temp_blob):
            send_ranks = range(1, self.splitted_size, 2)
	    #FIX this will not work properly for all configs
	    receive_ranks = range(0, self.splitted_size, 2)
            num_iter = int(math.ceil(math.log(self.splitted_size,2)))
            for i in range(num_iter):
                for (s_rank, r_rank) in zip(send_ranks, receive_ranks):
                    if self.splitted_rank == s_rank:
                        master_req = self.comm_splitted.Isend([ self.to_buffer(blob) ,MPI.FLOAT], dest=r_rank)
                        re=MPI.Request.Wait( master_req )
                    elif self.splitted_rank == r_rank:
                        slave_req = self.comm_splitted.Irecv([ self.to_buffer(temp_blob) ,MPI.FLOAT], source=s_rank)
                        re=MPI.Request.Wait( slave_req )
			cublas.cublasSaxpy(self.cublas_handle, temp_blob.size, 1.0, temp_blob.gpudata, 1, blob.gpudata, 1)
             	send_ranks = receive_ranks[1::2]
                receive_ranks = receive_ranks[::2]
	    	self.comm_splitted.Barrier()
		
	
	def initial_output( self ):
	    # Initial output
	    if self.is_global_master:
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


