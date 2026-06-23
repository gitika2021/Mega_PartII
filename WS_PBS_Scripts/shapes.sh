#!/bin/bash

ratio1=2
ratio2=3
seed=3

queue="debug"
walltime="00:05:00"
config_file="train_config_debug.json"

# queue="project"
# walltime="01:00:00"
# config_file="train_config.json"

log_dir="master_log_quick"
rsrp_dir="shapes"
base_dir="${log_dir}/${rsrp_dir}"

#config_file="train_config.json"
mkdir -p "${base_dir}"

export ratio1 ratio2 seed base_dir config_file

PBS_JOBNAME0="shape_gen"

# =========================
# Detect PBS availability
# =========================

if command -v qsub >/dev/null 2>&1; then
    echo "Running on HPC (PBS mode)"

    jid0=$(qsub -S /bin/bash \
        -v ratio1,ratio2,seed,base_dir,config_file \
        -q $queue \
        -N $PBS_JOBNAME0 \
        gene_shapes.sh)

else
    echo "Running locally (no PBS detected)"

    # run directly
    bash gene_shapes.sh
    
fi


# ratio1=3
# ratio2=3
# seed=50
# queue="debug"
# walltime="00:20:00"

# log_dir="master_log_quick"
# rsrp_dir="shapes"
# base_dir="${log_dir}/${rsrp_dir}"
# mkdir -p "${base_dir}"

# PBS_JOBNAME1="shape_gen"
# config_file="train_config_debug.json"
# jid1=$(qsub \
#     -q $queue \
#     -l walltime=$walltime \
#     -v ratio1=$ratio1,ratio2=$ratio2,seed=$seed,base_dir=$base_dir,config_file=$config_file \
#     -N $PBS_JOBNAME1 \
#     gene_shapes_quick.pbs)

# echo "Shape Generation Set 1 and Set 2 Job Submitted: $jid1"

