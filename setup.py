# Copyright (C) 2020 University of Colorado at Boulder
# This software may be modified and distributed under the terms of the
# MIT license. See the accompanying LICENSE file for details.
import os
import sys
from pathlib import Path
from shutil import rmtree, copytree
from setuptools import setup, find_packages
from setuptools.command.install import install
from subprocess import call
import pathlib
import pkg_resources
LIBRARY_NAME = "wombats"


with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]

## Installation of pygraphviz
# linux
if sys.platform == "linux" or sys.platform == "linux2":
    install_requires.append('pygraphviz')
    cmdclass = {}
# OS X
elif sys.platform == "darwin":
    class CustomInstall(install):
        """
        Somehow installing pygraphviz fails on mac.
        We have to tell pip where the graphviz libraries are on this machine.
        Plus, parse_requirements cannot handle arguments,
        so I had to use CustomInstall.
        """
        def run(self):
            call(['pip', 'install', '--global-option=build_ext', '--global-option="-I/usr/local/include"', '--global-option="-L/usr/local/lib"', 'pygraphviz'])
            install.run(self)
    cmdclass = {'install': CustomInstall}
# Windows...
elif sys.platform == "win32":
    raise Exception('Not Supported')

# Reference to the parents directory to find the packages correctly
src_path = Path(os.environ["PWD"], "../wombats")
dst_path = Path("./wombats")
copytree(src_path, dst_path)
setup(
    name=LIBRARY_NAME,
    version='1.0.0',
    author="Nicholas Renninger",
    author_email="nicholas.renninger@colorado.edu",
    description="",
    license="",
    url='https://github.com/aria-systems-group/wombats',
    python_requires='>=3.5',
    package_dir={"": "."},
    packages=find_packages(),
    cmdclass=cmdclass,
    install_requires=install_requires,
)
rmtree(dst_path)
