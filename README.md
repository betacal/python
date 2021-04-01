# Beta calibration - Python package

This package provides a BetaCalibration class which allows the user to fit any of our three proposed beta calibration models. For the paper and tutorials, check [https://betacal.github.io/](https://betacal.github.io/).

## Dependencies

* [Numpy] - NumPy is the fundamental package for scientific computing with
  Python.
* [Scikit-learn] - Machine Learning in Python.

## Usage

 - Install from pip using "pip install betacal"
 - Alternatively, download from the repository, cd to the folder and use "pip install ."
 - Once installed, import the package using "import betacal" 

## Unittest

Create a virtual environment with the necessary dependencies

```
virtualenv venv
. ./venv/bin/activate
pip install -r requirements.txt
```

and then run the script runtests.sh

```
bash runtests.sh
```

## License

MIT

[//]: # (References)
   [Numpy]: <http://www.numpy.org/>
   [Scikit-learn]: <http://scikit-learn.org/>
