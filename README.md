# Environment Setup

```
conda create -n crystal python=3.11
pip install -r requirements.txt
```

# Pipeline

Most of the intermediate files (besides CIF) are stored in `code/data/` as `.pkl` files.

## Data Query & Preprocess

Queries all materials from Materials Project API, and takes 1000 subset of each crystal system and convert to CIF. 

```
cd code
python get_mp_data.py
python convert_subset_to_cif.py
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
python vectorizers.py --landscapes
python vectorizers.py --image
```

Generated persistence landscapes/images are stored in `code/data/landscapes` and `data/images` as `.pkl` files (included in repo).