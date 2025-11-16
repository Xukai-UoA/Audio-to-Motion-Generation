import sys
sys.path.insert(0, '..')
"""
sys.path is a Python list that contains the paths that the Python interpreter searches for when importing modules.

- sys.path.insert(0, '..') 
Means: Insert the previous level directory of the current directory into the forefront of sys.path list(at index 0)

    - Python will prioritize searching the previous directory when importing modules.
    - If there are modules or packages in the higher-level directory, they will be loaded first, even if they have the same name as modules in the current path or other paths.
"""

from .dataUtils import *
from .common import *
from .skeleton import *
from .audio import *
from .text import *




