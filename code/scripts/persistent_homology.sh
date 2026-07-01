export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python compute_diagrams.py --source graph
python compute_diagrams.py --source point
python vectorizers.py --source graph --landscape --image
python vectorizers.py --source point --landscape --image

python compute_diagrams.py --abs
python vectorizers.py --abs --landscape --image

python compute_diagrams.py --plqy
python vectorizers.py --plqy --landscape --image

python compute_diagrams.py --plqy-full
python vectorizers.py --plqy-full --landscape --image