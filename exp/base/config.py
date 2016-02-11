FACADE_DIR = '../../'
FACADE_NAME = 'caffe_facade'
DSOLVER_DIR = '../../'
DSOLVER_NAME = 'dsolver'
SOLVER_CLASS = 'SGDSolver' # Possible Values: SGDSolver, NesterovSolver, AdaGradSolver
DSOLVER_CLASS = 'SDSolverMPI' # Possible Values: ADSolver, ADSolverMultiGPU, SDSolver, SDSolverMultiGPU
CONFIG_FILENAME = 'cifar10_full_multigpu_local.json' # Use 'cifar10_full_multigpu_sync.json' for synchronous training, and 'cifar10_full_multigpu_async.json' for asynchronous training
DSOLVER_MODE = 'multiGPU_MPI'

