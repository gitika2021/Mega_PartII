
ratio1=3
ratio2=3
seed=40

log_dir="master_log"
rsrp_dir="RsRp_${ratio1}_${ratio2}"
base_dir="${log_dir}/${rsrp_dir}"
mkdir -p "${base_dir}"

PBS_JOBNAME1="genlc_${ratio1}_${ratio2}"
PBS_JOBNAME2="train_${ratio1}_${ratio2}"
jid=$(qsub -v ratio1=$ratio1,ratio2=$ratio2,seed=$seed,base_dir=$base_dir \
    -N $PBS_JOBNAME1 \
    gene_data_debug.pbs)
echo "Light Curves Generated for radius ratio $ratio1 $ratio2"
basejid=$(echo $jid | cut -d'[' -f1)
echo "Job Id completed $basejid and training job is next"
qsub \
    -W depend=afterok:${basejid} \
    -v ratio1=$ratio1,ratio2=$ratio2,base_dir=$base_dir \
    -N $PBS_JOBNAME2 \
    train_data_debug.pbs

end_time=$(date +%s)
echo "Training Completed"
