import os


def open_write_file(dir_path, file_name):
    """Opens a file for writing, or creates new file if file doesn't exist."""
    file_path = os.path.join(dir_path, file_name)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    return file_path
