import pathlib
import sys
__version__ = '1.1.3'

sys.path.append('..\\')
from headpose.detect import PoseEstimator
DIR = pathlib.Path(__file__).parent.resolve()
