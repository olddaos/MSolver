ipcluster start -n 8 --profile=cifar10_full_multigpu_local &
sleep 8
ipython train.py
ipcluster stop --profile=cifar10_full_multigpu_local
