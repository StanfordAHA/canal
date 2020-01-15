#!/bin/bash

# force color
export PYTEST_ADDOPTS="--color=yes"

cd /canal/

pytest --pycodestyle  \
       --cov canal  \
       -v --cov-report term-missing tests
