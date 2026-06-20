import json
from mp_api.client import MPRester
from monty.json import MontyEncoder
# from monty.serialization import loadfn, dumpfn
from utils import CRYSTAL_SYSTEMS, FIELDS, open_write_file

from dotenv import load_dotenv
load_dotenv()


def save_doc_as_dict(doc) -> dict:
    json_object = dict()
    for field in FIELDS:
        json_object[field] = getattr(doc, field)
    return json_object


def query_all_crystals_from_mp() -> None:
    for system in CRYSTAL_SYSTEMS:
        print(f"Querying {system} system from Materials Project...")

        with MPRester(force_renew=True) as mpr:
            docs = mpr.materials.summary.search(
                crystal_system=system,
                fields=FIELDS
            )

        # serialize SummaryDoc into JSON
        # ! Note: directly using Monty serialization messes the material_id
        mp_json_dict = dict()
        for doc in docs:
            mp_json_dict[str(doc.material_id)] = save_doc_as_dict(doc)
        
        # save JSON files
        data_raw_dir = f'data/mp-raw'
        file_raw_name = f'{system}.json'
        data_raw_path = open_write_file(data_raw_dir, file_raw_name)

        with open(data_raw_path, 'w') as f:
            json.dump(mp_json_dict, f, cls=MontyEncoder, indent=4)


if __name__ == "__main__":
    query_all_crystals_from_mp()