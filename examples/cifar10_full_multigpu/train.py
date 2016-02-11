import sys
import config
import importlib
import socket
import os

sys.path.append(config.FACADE_DIR)
sys.path.append(config.DSOLVER_DIR)

facade = importlib.import_module(config.FACADE_NAME)
dsolver = importlib.import_module(config.DSOLVER_NAME)

dsolver_class = getattr(dsolver, config.DSOLVER_CLASS)
solver_class = getattr(facade, config.SOLVER_CLASS)

with dsolver_class(solver_class, config.CONFIG_FILENAME) as solver:
    solver.clean()
    solver.set_active_engines(config.DSOLVER_MODE)
    solver.init_engines()
    solver.deploy()
    solver.init_solver()
    solver.solve()
