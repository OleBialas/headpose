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
      url="https://github.com/OleBialas/headpose.git",
      author='Ole Bialas',
      author_email='bialas@cbs.mpg.de',
      license='MIT',
      python_requires='>=3.6',
      install_requires=['opencv-python',
                        'numpy',
                        'requests',
                        'torch',
                        'torchvision'],
      packages=find_packages(),
      package_data={'headpose': ['haarcascade_frontalface_default.xml']},
      include_package_data=True,
      zip_safe=False)

# add pypi.org/classifiers
