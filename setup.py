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
      url='https://github.com/pfriedrich-hub/headpose/headpose.git',
      author='Ole Bialas',
      author_email='bialas@cbs.mpg.de',
      license='MIT',
      python_requires='>=3.6',
      install_requires=['opencv-python',
                        'numpy'],
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False)

# add pypi.org/classifiers
