export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python compute_diagrams.py --source graph
python compute_diagrams.py --source point
python compute_diagrams.py --source custom
python vectorizers.py --source graph --landscape --image
python vectorizers.py --source point --landscape --image
python vectorizers.py --source custom --landscape --image

python compute_diagrams.py --source graph --abs
python compute_diagrams.py --source point --abs
python compute_diagrams.py --source custom --abs
python vectorizers.py --source graph --abs --landscape --image
python vectorizers.py --source point --abs --landscape --image
python vectorizers.py --source custom --abs --landscape --image