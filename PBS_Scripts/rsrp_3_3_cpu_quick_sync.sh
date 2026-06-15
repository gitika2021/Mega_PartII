
ratio1=3
ratio2=3
seed=4
# queue="debug"
# walltime="00:20:00"

queue="project"
walltime="00:20:00"

log_dir="master_log_quick"
rsrp_dir="RsRp_${ratio1}_${ratio2}"
base_dir="${log_dir}/${rsrp_dir}"
mkdir -p "${base_dir}"

PBS_JOBNAME0="gen_config_${ratio1}_${ratio2}"
PBS_JOBNAME1="genlc_${ratio1}_${ratio2}_s1"
PBS_JOBNAME2="genlc_${ratio1}_${ratio2}_s2"
PBS_JOBNAME3="genlc_${ratio1}_${ratio2}_s3"
# PBS_JOBNAME4="genlc_${ratio1}_${ratio2}_s4"
# PBS_JOBNAME5="genlc_${ratio1}_${ratio2}_s5"
PBS_JOBNAME6="train_${ratio1}_${ratio2}"

jid0=$(qsub -v ratio1=$ratio1,ratio2=$ratio2,seed=$seed,base_dir=$base_dir \
    -q $queue \
    -N $PBS_JOBNAME0 \
    gene_config_quick.pbs)

jid1=$(qsub \
    -q $queue \
    -l walltime=$walltime \
    -W depend=afterok:${jid0} \
    -v ratio1=$ratio1,ratio2=$ratio2,seed=$seed,base_dir=$base_dir \
    -N $PBS_JOBNAME1 \
    gene_data_set1.pbs)

jid2=$(qsub \
    -q $queue \
    -l walltime=$walltime \
    -W depend=afterok:${jid0} \
    -v ratio1=$ratio1,ratio2=$ratio2,seed=$seed,base_dir=$base_dir \
    -N $PBS_JOBNAME2 \
    gene_data_set2.pbs)

jid3=$(qsub \
    -q $queue \
    -l walltime=$walltime \
    -W depend=afterok:${jid0} \
    -v ratio1=$ratio1,ratio2=$ratio2,seed=$seed,base_dir=$base_dir \
    -N $PBS_JOBNAME3 \
    gene_data_set3.pbs)

qsub \
    -q $queue \
    -l walltime=$walltime \
    -W depend=afterok:${jid1}:${jid2}:${jid3} \
    -v ratio1=$ratio1,ratio2=$ratio2,base_dir=$base_dir \
    -N $PBS_JOBNAME6 \
    train_cpu.pbs

end_time=$(date +%s)
echo "Training Job Submitted"
