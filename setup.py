import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

about = {}
with open(os.path.join(here, 'pytorch_util', '__version__.py'), 'r') as f:
    exec(f.read(), about)

setup(
    name=about['__title__'],
    version=about['__version__'],
    description=about['__description__'],
    author=about['__author__'],
    author_email=about['__author_email__'],
    license=about['__license__'],
    include_package_data=True,
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        'numpy>=1.21.5',
        'pandas==1.2.4',
        'matplotlib==3.5.1',
        'matplotlib-inline==0.1.3',
        # 'ipython==8.4.0',
        'torch==1.11.0',
        'torchvision==0.12.0',
    ],
)
