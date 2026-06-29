export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

python get_mp_data.py
if [ ! -d "data/plqy/cif-plqy" ]; then
    echo "PLQY cif-plqy folder doesn't exist, unzipping..."
    tar -xzf cif-plqy.zip
fi
python get_mp_subset.py --seed --subset --size 5500 --large --size-large 7000 --abs --cif --plqy
