OMP_NUM_THREADS=8 \
MKL_NUM_THREADS=8 \
OPENBLAS_NUM_THREADS=8 \
taskset -c 32-63 bash scripts/dg_filter_mt.sh