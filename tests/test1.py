#scripts/script1.py

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_root)

import utils
