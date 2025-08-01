# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import setuptools
from setuptools import setup
import unittest


def my_test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('grid2op/tests', pattern='test_*.py')
    return test_suite


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

pkgs = {
    "required": [
        "scipy>=1.4.1",
        "pandas>=1.0.3",
        "pandapower>=3.1.1; python_version>='3.9'",
        "numpy",
        "scipy",
        "tqdm>=4.45.0",
        "networkx>=2.4",
        "requests>=2.23.0",
        "packaging",  # because gym changes the way it uses numpy prng in version 0.26 and i need both gym before and after...
        "typing_extensions",
        "orderly_set<5.4.0; python_version<='3.8'",
        "importlib_resources; python_version<='3.8'",
        "pandapower<3; python_version<='3.8'",
        "numpy<2; python_version<='3.8'",
        "scipy<1.14; python_version<='3.8'",
    ],
    "extras": {
        "optional": [
            "nbformat>=5.0.4",
            "jupyter-client>=6.1.0",
            "numba>=0.48.0",
            "matplotlib>=3.2.1",
            "plotly>=4.5.4",
            "seaborn>=0.10.0",
            "imageio>=2.8.0",
            "pygifsicle>=1.0.1",
            "psutil>=5.7.0",
            "gymnasium",
            "lightsim2grid",
        ],
        "gym": [
            "gym>=0.17.2",
        ],
        "gymnasium": [
            "gymnasium",
        ],
        "docs": [
            "numpydoc>=0.9.2",
            "sphinx<7.0.0,>=2.4.4",
            "sphinx-rtd-theme>=0.4.3",
            "sphinxcontrib-trio>=1.1.0",
            "autodocsumm>=0.1.13",
            "gym>=0.17.2",
            "gymnasium",
        ],
        "api": [
            "flask",
            "flask_wtf",
            "ujson"
        ],
        "plot": ["imageio"],
        "test": ["lightsim2grid",
                 "numba; python_version<='3.12'",  # numba not available on python 3.13 yet
                 "gymnasium",
                 "nbconvert",
                 "jinja2"
                 ],
        "chronix2grid": [
            "ChroniX2Grid>=1.2.0.post1",
            "pypsa<0.25"  # otherwise does not work (need fix in chronix2grid)
            ]
    }
}
pkgs["extras"]["test"] += pkgs["extras"]["optional"]
pkgs["extras"]["test"] += pkgs["extras"]["plot"]
pkgs["extras"]["test"] += pkgs["extras"]["gymnasium"]


setup(description='An gymnasium compatible environment to model sequential decision making for powersystems',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Benjamin DONNOT',
      author_email='benjamin.donnot@rte-france.com',
      python_requires='>=3.8',
      url="https://github.com/Grid2Op/grid2op",
      packages=setuptools.find_packages(),
      include_package_data=True,
      install_requires=pkgs["required"],
      extras_require=pkgs["extras"],
      zip_safe=False,
      entry_points={
          'console_scripts': [
              'grid2op.main=grid2op.command_line:main',
              'grid2op.download=grid2op.command_line:download',
              'grid2op.replay=grid2op.command_line:replay',
              'grid2op.testinstall=grid2op.command_line:testinstall'
          ]
      },
      test_suite='setup.my_test_suite'
      )
