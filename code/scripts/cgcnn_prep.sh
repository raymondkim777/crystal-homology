export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# python cgcnn_prep.py --abs --plqy --vector --bound
# python cgcnn_prep_s.py --abs --plqy --vector --bound

python cgcnn_prep.py --abs --vector --bound
python cgcnn_prep_s.py --abs --vector --bound