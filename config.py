FACADE_DIR = '/home/dlvr_solver_mesos'
FACADE_NAME = 'caffe_facade'
DSOLVER_DIR = '/home/dsolver_mesos'
DSOLVER_NAME = 'dsolver'
SOLVER_CLASS = 'SGDSolver' # Possible Values: SGDSolver, NesterovSolver, AdaGradSolver
DSOLVER_CLASS = 'MM_SDSolverMPI' # Possible Values: ADSolver, ADSolverMultiGPU, SDSolver, SDSolverMultiGPU
CONFIG_FILENAME = 'imagenet1000_sync.json' # Use 'cifar10_full_multigpu_sync.json' for synchronous training, and 'cifar10_full_multigpu_async.json' for asynchronous training
DSOLVER_MODE = 'multiGPU_MPI'

