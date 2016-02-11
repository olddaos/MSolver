#! /usr/bin/python
__author__ = 'akm'
from subprocess import call
import os
import sys

with open(os.path.join('', 'mesos_slave.out'), 'a') as out:
     out.write('subprocess called\n')
     out.write(os.getcwd()) 
call(["cp", os.path.join(sys.argv[1], "test_mpi.py"), os.getcwd()])
call(["cp", os.path.join(sys.argv[1], "launch_slave.sh"), os.getcwd()])
call(["cp", os.path.join(sys.argv[1], "dsolver.conf"), os.getcwd()])
call(["cp", os.path.join(sys.argv[1], "SimpleConfigParser.py"), os.getcwd()])
call(["./launch_slave.sh"])
