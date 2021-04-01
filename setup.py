from setuptools import setup
from distutils.util import convert_path
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

main_ns = {}
ver_path = convert_path('betacal/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setup(
    name='betacal',
    version=main_ns['__version__'],
    description='Beta calibration',
    author='Telmo de Menezes e Silva Filho and Miquel Perello Nieto',
    author_email='tmfilho@gmail.com',
    url = 'https://betacal.github.io/',
    download_url = 'https://github.com/betacal/python/archive/refs/tags/{}.tar.gz'.format(main_ns['__version__']),
    keywords = ['classifier calibration', 'calibration', 'classification'],
    license='MIT',
    packages=['betacal'],
    install_requires=[
        'numpy',
        'scikit-learn',
    ],
    long_description=long_description,
    long_description_content_type='text/markdown'
)
