#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import io
import os

from setuptools import setup, find_packages

# read the __version__ variable from nisqai/_version.py
exec(open("src/edd/_version.py").read())

# readme file as long description
long_description = ("======\n" +
                    "EDD\n" +
                    "======\n")
stream = io.open("README.md", encoding="utf-8")
stream.readline()
long_description += stream.read()

# read in requirements.txt
requirements = open("requirements.txt").readlines()
requirements = [r.strip() for r in requirements]

setup(
    name="edd",
    version=__version__,
    author="Nic Ezzell, Bruno Avritzer, Bibek Pokharel, Vinay Tripathi",
    author_email="nezzell@usc.edu",
    url="https://github.com/naezzell/edd-dev",
    description="Library for implementing dynamical decoupling schemes on NISQ-era quantum computers.",
    long_description=long_description,
    install_requires=requirements,
    license="Apache 2",
    packages=find_packages(where="src"),
    package_dir={"": "src"}
    )
