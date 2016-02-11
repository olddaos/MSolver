
n=4 # ADJUST
N=2 # ADJUST

exp_dir=$(pwd)
ssh dl02 'mkdir -p '$exp_dir
scp -q ./*.json ./*.py ./*.sh ./*.prototxt dsolver_machinefile $USER"@dl02:"$exp_dir"/"
echo "Experiment directory has been cloned to dl02: "$exp_dir

export PATH="/home/$USER/.openmpi/bin:$PATH"
export LD_LIBRARY_PATH="/home/$USER/.openmpi/lib/:$LD_LIBRARY_PATH"

CMD="nohup mpirun -x PATH --mca btl ^usnic,tcp -n "$n" -N "$N" -machinefile dsolver_machinefile python train.py"

echo "Running:"
echo $CMD" > log_out.log 2> log_err.log"
$CMD > log_out.log 2> log_err.log
