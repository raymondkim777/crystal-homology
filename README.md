# Environment Setup

```
conda create -n crystal python=3.11
pip install -r requirements.txt
```

# Pipeline

## Data Query & Preprocess

Queries all materials from Materials Project API, and takes an optimal 7000 subset of each crystal system. By default, also queries absorption data and saves separately. Options can be changed to merge absorption data with main subset, or not query absorption entirely. Converts all crystal structures to CIF. 

```
cd code
bash scripts/data_preprocess.sh
```

Raw Materials Project data is stored in `code/data/mp-raw`, main subset is stored in `code/data/pretrain/mp-subset`, and converted CIF files are stored in `code/data/pretrain/cif`. By default, absorption data is separately but identically stored in `code/data/abs`. 

## Atomic Bond Network Creation

Uses CrystalNN to compute atomic bonds and create NetworkX digraphs (also stores undirected versions). Also computes graph bounds (max atomic bond distance/neighbor count) for later use. Repeats for both main subset and absorption data. 

```
bash scripts/make_graph.sh
```

Generated graphs are stored in `code/data/<folder>/graphs-multi` and `code/data/<folder>/graphs` as `.pkl` files (included in repo), where `<folder>` is `pretrain` or `abs`. 

## Persistence Diagram/Vectorizations Generation

Generates persistence diagrams, landscapes, and images for all crystals. 

```
bash scripts/persistent_homology.sh
```

Generated persistence diagrams, landscapes, and images are stored in `code/data/<folder>/diagrams`, `code/data/<folder>/landscapes` and `code/data/<folder>/images` as `.pkl` files, where `<folder>` is `pretrain` or `abs`. Also computes persistence image bounds for later, for `GraphData` input.

## CGCNN Prep

Moves graph/vectorization data over to `cgcnn/data/<folder>` folder, and stores `id_prop.csv` and `id_mask.csv`, where `<folder>` is `pretrain` or `abs`. 

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