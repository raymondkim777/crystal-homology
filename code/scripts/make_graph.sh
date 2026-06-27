export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python create_bonds.py
python find_bounds.py --bounds --stats

python create_bonds.py --abs
python find_bounds.py --abs --bounds --stats