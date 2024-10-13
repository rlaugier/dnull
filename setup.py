import sys
from setuptools import setup

setup(name="dnull",
    version="0.0.1",
    description="A framework to handle the NIFITS Nullin Interferometry data standard",
    url="--",
    author="Romain Laugier",
    author_email="romain.laugier@kuleuven.be",
    license="BSD-3-Clause",
    classifiers=[
      'Development Status :: 2 - Pre-alpha',
      'Intended Audience :: Professional Astronomers',
      'Topic :: High Angular Resolution Astronomy :: Interferometry :: High-contrast',
      'Programming Language :: Python :: 3.10'
    ],
    packages=["dnull"],
    install_requires=[
            'numpy', 'scipy', 'matplotlib', 'astropy', "nifits"
            'sympy', 'einops', "jax", "zodiax"
    ],
    zip_safe=False)
