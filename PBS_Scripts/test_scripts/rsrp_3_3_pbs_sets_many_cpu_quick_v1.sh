
ratio1=3
ratio2=3
seed=4

log_dir="master_log_quick"
rsrp_dir="RsRp_${ratio1}_${ratio2}"
base_dir="${log_dir}/${rsrp_dir}"
mkdir -p "${base_dir}"

PBS_JOBNAME1="genlc_${ratio1}_${ratio2}_s1"
PBS_JOBNAME2="genlc_${ratio1}_${ratio2}_s2"
PBS_JOBNAME3="genlc_${ratio1}_${ratio2}_s3"
PBS_JOBNAME4="genlc_${ratio1}_${ratio2}_s4"
PBS_JOBNAME5="genlc_${ratio1}_${ratio2}_s5"
PBS_JOBNAME6="train_${ratio1}_${ratio2}"

jid1=$(qsub -v ratio1=$ratio1,ratio2=$ratio2,seed=$seed,base_dir=$base_dir \
    -N $PBS_JOBNAME1 \
    gene_data_set1_quick.pbs)

echo "Light Curve Generation Set 1 Job Submitted: $jid1"

jid2=$(qsub -v ratio1=$ratio1,ratio2=$ratio2,seed=$seed,base_dir=$base_dir \
    -N $PBS_JOBNAME2 \
    gene_data_set2_quick.pbs)

echo "Light Curve Generation Set 2 Job Submitted: $jid2"

jid3=$(qsub -v ratio1=$ratio1,ratio2=$ratio2,seed=$seed,base_dir=$base_dir \
    -N $PBS_JOBNAME3 \
    gene_data_set3_quick.pbs)

jid4=$(qsub -v ratio1=$ratio1,ratio2=$ratio2,seed=$seed,base_dir=$base_dir \
    -N $PBS_JOBNAME4 \
    gene_data_set4_quick.pbs)

jid5=$(qsub -v ratio1=$ratio1,ratio2=$ratio2,seed=$seed,base_dir=$base_dir \
    -N $PBS_JOBNAME5 \
    gene_data_set5_quick.pbs)


qsub \
    -W depend=afterok:${jid1}:${jid2}:${jid3}:${jid4}:${jid5} \
    -v ratio1=$ratio1,ratio2=$ratio2,base_dir=$base_dir \
    -N $PBS_JOBNAME6 \
    train_data_quick.pbs

end_time=$(date +%s)
echo "Training Job Submitted"
