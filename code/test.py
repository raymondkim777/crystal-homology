import os

dir_path = "data/cif/hexagonal"

file_count = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])

print(f"Number of files: {file_count}")