```
conda create -n crystal python=3.11
pip install -r requirements.txt
```

# Data Query & Preprocess

```
cd code
python get_mp_data.py
python convert_subset_to_cif.py
```

Queried CIF files are stored in `data/cif`.

# Atomic Bond Network Creation

```
python create_bonds.py
```

Generated graphs are stored in `data/graphs` as .pkl files (included in git repo).

# Persistence Diagram Creation

```
python compute_diagrams.py
```

Generated persistence diagrams are stored in `data/diagrams` as .pkl files (included in git repo).

# Vectorizations

(Vectorization functions for landscape/image (using giotto-tda) is pushed, but automated code is not.)