import json
import os


def read_json_file(file_path: str) -> dict:
    with open(file_path) as file:
        data = json.load(file)
    return data


def create_directory(path):
    if not os.path.exists(path):
        print("Creating directories for path: {}".format(path))
        os.makedirs(path)
    else:
        print("Already exists path: {}".format(path))
