import os
import sys


# TODO: this sucks
COMFY_PATH = os.path.realpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")
)
sys.path.insert(0, COMFY_PATH)

from folder_paths import (
    models_dir,
    get_output_directory,
    get_temp_directory,
    get_save_image_path,
)
