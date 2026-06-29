export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python create_bonds.py
python find_bounds.py --bounds --avail --dist

python create_bonds.py --abs
python find_bounds.py --abs --bounds --avail --dist

python create_bonds.py --plqy
python find_bounds.py --plqy --bounds