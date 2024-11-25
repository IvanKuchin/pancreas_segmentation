import os
import sys

def setup():
    project_folder = os.path.dirname(os.path.dirname(__file__))
    src_folder = os.path.join(project_folder, "src")
    sys.path.append(src_folder)
