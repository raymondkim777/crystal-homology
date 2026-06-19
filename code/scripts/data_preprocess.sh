export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# python get_mp_data.py
python convert_subset_to_cif.py --subset --size 6700 --absorb --cif

rm -f data/cif_files.zip
python -m zipfile -c data/cif_files.zip data/cif/