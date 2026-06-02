from mp_api.client import MPRester
from dotenv import load_dotenv


load_dotenv()


with MPRester() as mpr:
    # docs = mpr.materials.summary.search(
    #     material_ids=["mp-13"]
    # )
    structure = mpr.get_structure_by_material_id("mp-149")
    print(structure)

# with open('test_data.txt', 'w', encoding='utf-8') as f:
#     f.write(str(docs[0]))


# with MPRester(monty_decode=False, use_document_model=False) as mpr:
#     pass