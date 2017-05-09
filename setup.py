from setuptools import setup

setup(name='betacal',
      version='0.2.5',
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
