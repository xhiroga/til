import os

script_dir = os.path.dirname(__file__)


def get_script(rel_path):
    abs_file_path = os.path.join(script_dir, rel_path)
    with open(abs_file_path) as file:
        return file.read()
