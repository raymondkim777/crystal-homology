from mp_api.client import MPRester
from dotenv import load_dotenv
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


load_dotenv()


# 1. Load a crystal structure (e.g., from a CIF file or by importing from your dataset)
with MPRester() as mpr:
    docs = mpr.materials.summary.search(
        material_ids=["mp-149"]
    )
    structure = mpr.get_structure_by_material_id("mp-149")

print(docs[0].symmetry.crystal_system)
print(docs[0].possible_species)

# # 2. Initialize the symmetry analyzer
# finder = SpacegroupAnalyzer(structure)

# crystal_system = finder.get_crystal_system()

# print(f"Crystal System: {crystal_system}")