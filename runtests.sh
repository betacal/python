#!/bin/bash

echo "Activating virtual environment"
. ./venv/bin/activate

# with coverage it tests sklearn as well
# nosetests --with-coverage
nosetests
