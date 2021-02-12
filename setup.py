from setuptools import setup, find_packages
import re

with open('README.md') as f:
    readme = f.read()

# extract version
with open('headpose/__init__.py') as file:
    for line in file.readlines():
        m = re.match("__version__ *= *['\"](.*)['\"]", line)
        if m:
            version = m.group(1)

setup(name='headpose',
      version=version,
      description='estimate the pose of the head based on an image.',
      long_description=readme,
      long_description_content_type='text/markdown',
      url='https://github.com/OleBialas/headpose.git',
      author='Ole Bialas',
      author_email='bialas@cbs.mpg.de',
      license='MIT',
      python_requires='>=3.6',
      install_requires=['tensorflow>2, <2.4',
                        'opencv-python',
                        'numpy'],
      packages=find_packages(),
      package_data={'headpose': ['model/caffemodel',
                                 'model/prototxt',
                                 'model/model_points.txt',
                                 'model/pose_model/saved_model.pb',
                                 'model/pose_model/variables/*',
                                 'model/pose_model/variables/variables.variables.data-00000-of-00001',
                                 'model/pose_model/variables/variables.index']},
      include_package_data=True,
      zip_safe=False)

# add pypi.org/classifiers
