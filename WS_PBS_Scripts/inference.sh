#!/bin/bash

ratio1=2
ratio2=3
seed=3
snr_min=100
snr_max=500
noise="gaussian"
train_network="curriculam_noise"

noise="real"
train_network="fixed_noise"

nproc=24 # updated for local machine


queue="debug"
walltime="00:15:00"

# queue="project"
# walltime="01:00:00"

# config_file="train_config_debug.json"
# log_dir="master_log_quick"

config_file="train_config.json"
log_dir="master_log"

rsrp_dir="RsRp_${ratio1}_${ratio2}"
base_dir="${log_dir}/${rsrp_dir}"
mkdir -p "${base_dir}"

export ratio1 ratio2 seed base_dir config_file snr_min snr_max noise train_network nproc

PBS_JOBNAME0="inference_${ratio1}_${ratio2}"
# =========================
# Detect PBS availability
# =========================

if command -v qsub >/dev/null 2>&1; then
    echo "Running on HPC (PBS mode)"

    jid0=$(qsub -S /bin/bash \
        -q $queue \
        -l walltime=$walltime \
        -v ratio1,ratio2,seed,base_dir,config_file,snr_min,snr_max,noise,train_network \
        -N $PBS_JOBNAME0 \
        gene_inference.sh)
else
    echo "Running locally (no PBS detected)"

    # run directly
    bash gene_inference.sh
    
fi
