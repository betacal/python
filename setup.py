from setuptools import setup
from distutils.util import convert_path

main_ns = {}
ver_path = convert_path('betacal/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setup(name='betacal',
      version=main_ns['__version__'],
      description='Beta calibration',
      url='https://github.com/betacal/python',
      author='tmfilho',
      author_email='tmfilho@gmail.com',
      license='MIT',
      packages=['betacal'],
      install_requires=[
          'numpy',
          'sklearn',
      ],
      zip_safe=False)
