#!/bin/bash

ratio1=2
ratio2=3
seed=3

queue="debug"
walltime="00:20:00"
config_file="train_config_debug.json"

# queue="project"
# walltime="01:00:00"
# config_file="train_config.json"

log_dir="master_log_quick"
rsrp_dir="RsRp_${ratio1}_${ratio2}"
base_dir="${log_dir}/${rsrp_dir}"
mkdir -p "${base_dir}"

export ratio1 ratio2 seed base_dir config_file

PBS_JOBNAME0="gen_config_${ratio1}_${ratio2}"
PBS_JOBNAME1="genlc_${ratio1}_${ratio2}_s1"
PBS_JOBNAME2="genlc_${ratio1}_${ratio2}_s2"
PBS_JOBNAME3="genlc_${ratio1}_${ratio2}_s3"
PBS_JOBNAME6="train_${ratio1}_${ratio2}"
# =========================
# Detect PBS availability
# =========================

if command -v qsub >/dev/null 2>&1; then
    echo "Running on HPC (PBS mode)"

    jid0=$(qsub -S /bin/bash \
        -v ratio1,ratio2,seed,base_dir,config_file \
        -q $queue \
        -N $PBS_JOBNAME0 \
        gene_config.sh)

    # jid1=$(qsub -S /bin/bash \
    #     -q $queue \
    #     -l walltime=$walltime \
    #     -W depend=afterok:${jid0} \
    #     -v ratio1,ratio2,seed,base_dir \
    #     -N $PBS_JOBNAME1 \
    #     gene_data_set1.sh)

    jid6=$(qsub -S /bin/bash \
        -q $queue \
        -l walltime=$walltime \
        -W depend=afterok:${jid0} \
        -v ratio1,ratio2,seed,base_dir \
        -N $PBS_JOBNAME6 \
        train_cpu.sh)

else
    echo "Running locally (no PBS detected)"

    # run directly
    bash gene_config.sh
    bash gene_data_set1.sh
    bash train_cpu.sh
fi