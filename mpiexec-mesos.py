#!/usr/bin/env python

import mesos.interface
import mesos.native
from mesos.interface import mesos_pb2
import os
import sys
import time
import re
import threading

from optparse import OptionParser
from subprocess import *

import SimpleConfigParser
parser_config = SimpleConfigParser.SimpleConfigParser()

class MPIScheduler(mesos.interface.Scheduler):

  def __init__(self, options, ip, port):
    self.mpdsLaunched = 0
    self.mpdsFinished = 0
    self.ip = ip
    self.port = port
    self.options = options
    self.startedExec = False

  def registered(self, driver, fid, masterInfo):
    print "Mesos MPI scheduler and mpd running at %s:%s" % (self.ip, self.port)
    print "Registered with framework ID %s" % fid.value

  def resourceOffers(self, driver, offers):
    print "Got %d resource offers" % len(offers)

    for offer in offers:
      print "Considering resource offer %s from %s" % (offer.id.value, offer.hostname)

      if self.mpdsLaunched > TOTAL_MPDS:
        print "Declining permanently because we have already launched enough tasks"
        driver.declineOffer(offer.id)
        continue

      cpus = 0
      mem = 0
      tasks = []

      for resource in offer.resources:
        if resource.name == "cpus":
          cpus = resource.scalar.value
        elif resource.name == "mem":
          mem = resource.scalar.value

      if cpus < CPUS or mem < MEM:
        print "Declining offer due to too few resources"
        driver.declineOffer(offer.id)
      else:
        tid = self.mpdsLaunched
        self.mpdsLaunched += 1

        print "Accepting offer on %s to start mpd %d" % (offer.hostname, tid)

        task = mesos_pb2.TaskInfo()
        task.task_id.value = str(tid)
        task.slave_id.value = offer.slave_id.value
        task.name = "task %d " % tid

        cpus = task.resources.add()
        cpus.name = "cpus"
        cpus.type = mesos_pb2.Value.SCALAR
        cpus.scalar.value = CPUS

        mem = task.resources.add()
        mem.name = "mem"
        mem.type = mesos_pb2.Value.SCALAR
        mem.scalar.value = MEM

        uri = task.command.uris.add()
        uri.value= parser_config.get_option("PATH_TO_DEPLOY_SCRIPT")

        #task.command.value = "python ./deploy_slave.py /mnt/resource_manager_new/" #parser_config.get_option("TASK_SLAVE_COMMAND").strip('"')

        task.command.value = "nohup mpirun --am dsolver.mca -n 8 -N 4 -machinefile dsolver_machinefile python train.py > training_withmca_4x4.out 2> training_withmca_4x4.err &" #parser_config.get_option("TASK_SLAVE_COMMAND").strip('"')

        tasks.append(task)

        print "Replying to offer: launching mpd %d on host %s" % (tid, offer.hostname)
        driver.launchTasks(offer.id, tasks)


  def statusUpdate(self, driver, update):
    print "Task %s in state %s" % (update.task_id.value, update.state)
    if (update.state == mesos_pb2.TASK_FAILED or
        update.state == mesos_pb2.TASK_KILLED or
        update.state == mesos_pb2.TASK_LOST):
      print "A task finished unexpectedly, calling mpdexit" 
      driver.stop()
    if (update.state == mesos_pb2.TASK_FINISHED):
      self.mpdsFinished += 1
      if self.mpdsFinished == TOTAL_MPDS:
        print "All tasks done, all mpd's closed, exiting"
        driver.stop()


if __name__ == "__main__":
  global parser_config

  parser = OptionParser(usage="Usage: %prog [options] mesos_master mpi_program")
  parser.disable_interspersed_args()
  parser.add_option("-n", "--num",
                    help="number of mpd's to allocate (default 1)",
                    dest="num", type="int", default=1)
  parser.add_option("-c", "--cpus",
                    help="number of cpus per mpd (default 1)",
                    dest="cpus", type="int", default=1)
  parser.add_option("-m","--mem",
                    help="number of MB of memory per mpd (default 1GB)",
                    dest="mem", type="int", default=1024)
  parser.add_option("--name",
                    help="framework name", dest="name", type="string")
  parser.add_option("-p","--path",
                    help="path to look for MPICH2 binaries (mpd, mpiexec, etc.)",
                    dest="path", type="string", default="")
  parser.add_option("--ifhn-master",
                    help="alt. interface hostname for what mpd is running on (for scheduler)",
                    dest="ifhn_master", type="string")

  # Add options to configure cpus and mem.
  (options,args) = parser.parse_args()
  if len(args) < 1:
    print >> sys.stderr, "At config file is required."
    print >> sys.stderr, "Use --help to show usage."
    exit(2)

  TOTAL_MPDS = options.num
  CPUS = options.cpus
  MEM = options.mem

  parser_config.read( args[0] )
  print "Launching Mesos using config %s" % args[0]

  ip = "127.0.0.1"
  port = "5050"
  scheduler = MPIScheduler(options, ip, port)

  framework = mesos_pb2.FrameworkInfo()
  framework.user = ""

  framework.name = "DsolverMPI"

  driver = mesos.native.MesosSchedulerDriver(
    scheduler,
    framework,
    '10.51.177.50:5050')
  sys.exit(0 if driver.run() == mesos_pb2.DRIVER_STOPPED else 1)
