# Environment Setup

```
conda create -n crystal python=3.11
pip install -r requirements.txt
```

# Pipeline

Most of the intermediate files (besides CIF) are stored in `code/data/` as `.pkl` files.

## Data Query & Preprocess

Queries all materials from Materials Project API, and takes 1000 subset of each crystal system and converts to CIF. 

```
cd code
python get_mp_data.py
python convert_subset_to_cif.py --subset-size 6700
```

Queried CIF files are stored in `code/data/cif` (not in repo).

## Atomic Bond Network Creation

Uses CrystalNN to compute atomic bonds and create NetworkX graph. 

```
python create_bonds.py
```

Generated graphs are stored in `code/data/graphs` as `.pkl` files (included in repo).

## Persistence Diagram Creation

Generates persistence diagrams for all 7000 crystals. 

```
python compute_diagrams.py
```

Generated persistence diagrams are stored in `code/data/diagrams` as `.pkl` files (included in repo).

## Vectorizations

Generates persistence landscapes and images for all 7000 crystals.

Add `--example` to plot the persistence diagram and landscape/image of an example material. (Edit code directly to select specific system, material, and dimensions to plot.)

```
python vectorizers.py --landscape
python vectorizers.py --image
```

Generated persistence landscapes/images are stored in `code/data/landscapes` and `data/images` as `.pkl` files (included in repo).

## CGCNN Prep

Moves graph data over to `cgcnn/data/graph_data` folder as `.pkl` file and creates `id_prop.csv`. Also computes dataset bounds required as input for `GraphData` class.

```
python cgcnn_prep.py --save
```
If `--save` argument is not included, then `graph_data` and `id_prop.csv` are not created.

## CGCNN Training

Trains CGCNN to predict crystal systems via classification (first pass). 

```
cd cgcnn
```

```
python main.py --seed --task classification --optim Adam --num-classes 7 --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --start-epoch 100 --lr 0.001 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 data/our-data
```