# Infer the predicted shapes for various radius ratio bins on test and Kepler Light Curves
#=========================================================================================
#queue="debug"
#walltime="00:20:00"

queue="project"
walltime="24:00:00"
#=================================================================
log_dir="master_log"
#=================================================================

ratio1=10
ratio2=12
seed=11

rsrp_dir="RsRp_${ratio1}_${ratio2}"
base_dir="${log_dir}/${rsrp_dir}"
mkdir -p "${base_dir}"

PBS_JOBNAME="inference_${ratio1}_${ratio2}"
jid0=$(qsub -v ratio1=$ratio1,ratio2=$ratio2,seed=$seed,base_dir=$base_dir \
    -q $queue \
    -N $PBS_JOBNAME \
    inference.pbs)

#=================================================================



#=================================================================



end_time=$(date +%s)
echo "Training Job Submitted"
