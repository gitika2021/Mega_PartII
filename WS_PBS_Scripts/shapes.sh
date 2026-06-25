#!/bin/bash

ratio1=2
ratio2=3
seed=3

queue="debug"
walltime="00:05:00"

# queue="project"
# walltime="01:00:00"

config_file="train_config_debug.json"
log_dir="master_log_quick"

# config_file="train_config.json"
#log_dir="master_log"

shape_dir="shapes"
base_dir="${log_dir}/${shape_dir}"

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
