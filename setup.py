from setuptools import setup

setup(name='betacal',
      version='0.2.4',
      description='Beta calibration',
      url='https://github.com/betacal',
      author='tmfilho',
      author_email='tmfilho@gmail.com',
      license='MIT',
      packages=['betacal'],
      install_requires=[
          'numpy',
          'sklearn',
      ],
      zip_safe=False)
