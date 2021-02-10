import pathlib
import sys
__version__ = '0.1'

sys.path.append('..\\')
DIR = pathlib.Path(__file__).parent.resolve()

from headpose import PoseEstimator
