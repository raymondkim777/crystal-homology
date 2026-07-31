# Environment Setup

```
conda create -n crystal python=3.11
pip install -r requirements.txt
```

# Pipeline

## Data Query & Preprocess

(Tar file `code/data/data.tar.gz` is provided for submission, containing all outputs from this section.)

Queries all materials from Materials Project API, and takes an optimal 7000 subset of each crystal system. Queries absorption data and saves separately. Converts all crystal structures to CIF. 

NOTE: Materials Project API key must be stored in `.env` file under `MP_API_KEY` before running. 

```
cd code
bash scripts/data_preprocess.sh
```

Raw Materials Project data is stored in `code/data/mp-raw`. Subset files are stored in `code/data/pretrain/`, with MP SummaryDocs stored in `code/data/mp-subset`. Absorption data is separately stored in `code/data/abs`, with MP SummaryDocs in `code/data/mp-abs`. 

## Atomic Bond Network Creation

Uses CrystalNN to compute atomic bonds and create NetworkX digraphs (directed and undirected) from CIF files. Also computes graph bounds (max atomic bond distance/neighbor count) for later use. Repeats for both main subset and absorption data. 

```
bash scripts/make_graph.sh
```

Generated graphs are stored in `code/data/<folder>/graphs-multi` and `code/data/<folder>/graphs` as `.pkl` files, where `<folder>` is `pretrain` or `abs`. 

## Persistence Diagram/Vectorizations Generation

Generates persistence diagrams, landscapes, and images for all crystals, for both graphs and point clouds. 

```
bash scripts/persistent_homology.sh
```

Generated persistence diagrams, landscapes, and images are stored in `code/data/<folder>/diagrams`, `code/data/<folder>/landscapes` and `code/data/<folder>/images` as `.pkl` files, where `<folder>` is `pretrain` or `abs`. Also computes persistence image bounds for later, for `GraphData` input.

## TopoMT-CGCNN Prep

Moves graph/vectorization data over to `cgcnn/data/<folder>` folder, and stores `id_prop.csv` and `id_mask.csv`, where `<folder>` is `pretrain` or `abs`. 

```
bash scripts/cgcnn_prep.sh
```

## TopoMT-CGCNN Training

Trains CGCNN+ to predict crystal systems via classification, with vectorization options. Example command listed below for training a model with landscape vectorizations of graph PH features, using graph augmentation with cartesian displacement vectors. 

Note: GraphData parameters `max_num_nbr` and `dmax` need to be set to the smallest integers greater than or equal to the corresponding numbers in `data/<folder>/tasks/bounds.json`.

```
cd cgcnn
```
```
python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 150 --lr 0.001 --batch-size 256 --norm-sample 0 --id ex_graph_landscape --attr --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source graph --vector landscape --train pretrain data/pretrain
```