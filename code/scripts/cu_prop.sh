export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# if [ ! -d "data/plqy/cif-plqy" ]; then
#     echo "PLQY cif-plqy folder doesn't exist, unzipping..."
#     unzip data/cif-plqy.zip -d data/plqy
# fi
# python get_mp_subset.py --ple



export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python create_bonds.py --plqy
python find_bounds.py --plqy --bounds



python compute_diagrams.py --source graph --plqy
python compute_diagrams.py --source point --plqy
python compute_diagrams.py --source custom --plqy
python vectorizers.py --source graph --plqy --landscape --image
python vectorizers.py --source point --plqy --landscape --image
python vectorizers.py --source custom --plqy --landscape --image
