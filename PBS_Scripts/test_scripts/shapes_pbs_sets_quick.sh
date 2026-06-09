
ratio1=3
ratio2=3
seed=50

log_dir="master_log_quick"
rsrp_dir="shapes"
base_dir="${log_dir}/${rsrp_dir}"
mkdir -p "${base_dir}"

PBS_JOBNAME1="shape_gen"

jid1=$(qsub -v ratio1=$ratio1,ratio2=$ratio2,seed=$seed,base_dir=$base_dir \
    -N $PBS_JOBNAME1 \
    gene_shapes_set1_set2_quick.pbs)

echo "Shape Generation Set 1 and Set 2 Job Submitted: $jid1"

