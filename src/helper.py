import os


def get_file_path_in_project(dir_name: str, file_name: str) -> str:
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_dir, dir_name, file_name)
