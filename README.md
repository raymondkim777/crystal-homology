# Environment Setup

```
conda create -n crystal python=3.11
pip install -r requirements.txt
```

# Pipeline

Most of the intermediate files (besides CIF) are stored in `code/data/` as `.pkl` files.

## Data Query & Preprocess

Queries all materials from Materials Project API, and takes an optimal 6700 subset of each crystal system. Queries absorption data and combines with subset. Converts all crystal structures to CIF. 

```
cd code
bash scripts/data_preprocess.sh
```

Raw Materials Project data is stored in `code/data/mp-raw`, subset is stored in `code/data/mp-subset`, and converted CIF files are stored in `code/data/cif` (not in repo).

## Atomic Bond Network Creation

Uses CrystalNN to compute atomic bonds and create NetworkX digraphs (also stores undirected versions). Also need to compute bounds for future input. 

```
bash scripts/make_graph.sh
```

Generated graphs are stored in `code/data/graphs-multi` and `code/data/graphs` as `.pkl` files (included in repo). Add `--test` argument to `find_bounds.py` to check for bidirectionality in directed graphs.

## Persistence Diagram/Vectorizations Generation

Generates persistence diagrams, landscapes, and images for all crystals. 

```
bash scripts/persistent_homology.sh
```

Generated persistence diagrams are stored in `code/data/diagrams` as `.pkl` files (included in repo). Generated persistence landscapes/images are stored in `code/data/landscapes` and `data/images` as `.pkl` files (included in repo).

## CGCNN Prep

Moves graph/vectorization data over to `cgcnn/data/graph_data` folder as `.pkl` file and creates `id_prop.csv`. Also computes dataset bounds required as input for `GraphData` class.

```
bash scripts/cgcnn_prep.sh

```

## CGCNN Training

Trains CGCNN+ to predict crystal systems via classification, with vectorization options. 

```
cd cgcnn
```
```
bash train.sh
```