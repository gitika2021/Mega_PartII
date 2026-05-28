
ratio1=12
ratio2=15
seed=60

log_dir="master_log"
rsrp_dir="RsRp_${ratio1}_${ratio2}"
base_dir="${log_dir}/${rsrp_dir}"
mkdir -p "${base_dir}"

PBS_JOBNAME1="genlc_${ratio1}_${ratio2}"
PBS_JOBNAME2="train_${ratio1}_${ratio2}"

jid1=$(qsub -v ratio1=$ratio1,ratio2=$ratio2,seed=$seed,base_dir=$base_dir \
    -N $PBS_JOBNAME1 \
    gene_data_set1.pbs)

echo "Light Curve Generation Set 1 Job Submitted: $jid1"

jid2=$(qsub -v ratio1=$ratio1,ratio2=$ratio2,seed=$seed,base_dir=$base_dir \
    -N $PBS_JOBNAME1 \
    gene_data_set2.pbs)

echo "Light Curve Generation Set 2 Job Submitted: $jid2"

qsub \
    -W depend=afterok:${jid1}:${jid2} \
    -v ratio1=$ratio1,ratio2=$ratio2,base_dir=$base_dir \
    -N $PBS_JOBNAME2 \
    train_data.pbs

end_time=$(date +%s)
echo "Training Job Submitted"
