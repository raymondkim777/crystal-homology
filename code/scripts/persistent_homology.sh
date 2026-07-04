export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# python compute_diagrams.py --source graph
# python compute_diagrams.py --source point
# python vectorizers.py --source graph --landscape --image
# python vectorizers.py --source point --landscape --image

# python compute_diagrams.py --source graph --abs
# python compute_diagrams.py --source point --abs
# python vectorizers.py --source graph --abs --landscape --image
# python vectorizers.py --source point --abs --landscape --image

# python compute_diagrams.py --source graph --plqy
# python compute_diagrams.py --source point --plqy
# python vectorizers.py --source graph --plqy --landscape --image
# python vectorizers.py --source point --plqy --landscape --image

python compute_diagrams.py --source graph --plqy-full
python compute_diagrams.py --source point --plqy-full
python vectorizers.py --source graph -plqy-full --landscape --image
python vectorizers.py --source point -plqy-full --landscape --image