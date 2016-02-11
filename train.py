import sys
import config
import importlib
import socket
import os

#print "I start"
sys.path.insert(0, config.FACADE_DIR)
sys.path.insert(0, config.DSOLVER_DIR)


print sys.path
facade = importlib.import_module(config.FACADE_NAME)
dsolver = importlib.import_module(config.DSOLVER_NAME)

dsolver_class = getattr(dsolver, config.DSOLVER_CLASS)
solver_class = getattr(facade, config.SOLVER_CLASS)
#print "Here"
with dsolver_class(solver_class, config.CONFIG_FILENAME) as solver:
    solver.clean()
    #solver.set_active_engines(config.DSOLVER_MODE)
    solver.init_engines()
    solver.deploy()
    solver.init_solver()
    solver.solve()

