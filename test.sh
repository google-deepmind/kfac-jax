#!/bin/bash
# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -e

readonly VENV_DIR=/tmp/kfac-jax-test-env
echo "Creating virtual environment under ${VENV_DIR}."
echo "You might want to remove this when you no longer need it."

# Install deps in a virtual env.
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python --version

python -m pip install --upgrade pip setuptools

# Run setup.py, this installs the python dependencies
python -m pip install .[tests]

# Run tests using pytest.
python -m pytest -n 2 tests/
