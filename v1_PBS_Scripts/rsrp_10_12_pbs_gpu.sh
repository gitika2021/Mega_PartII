
ratio1=10
ratio2=12
seed=50

log_dir="master_log"
rsrp_dir="RsRp_${ratio1}_${ratio2}"
base_dir="${log_dir}/${rsrp_dir}"
mkdir -p "${base_dir}"

PBS_JOBNAME1="genlc_${ratio1}_${ratio2}"
PBS_JOBNAME2="train_${ratio1}_${ratio2}"
jid=$(qsub -v ratio1=$ratio1,ratio2=$ratio2,seed=$seed,base_dir=$base_dir \
    -N $PBS_JOBNAME1 \
    gene_data.pbs)

basejid=$(echo $jid | cut -d'[' -f1)

qsub \
    -W depend=afterok:${basejid} \
    -v ratio1=$ratio1,ratio2=$ratio2,base_dir=$base_dir \
    -N $PBS_JOBNAME2 \
    train_data_gpu.pbs

end_time=$(date +%s)

