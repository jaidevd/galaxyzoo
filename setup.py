from setuptools import setup, find_packages

NAME = "galaxyzoo"

setup(
    name=NAME,
    version='0.0.1',
    author='Jaidev Deshpande',
    author_email='deshpande.jaidev@gmail.com',
    packages=['galaxyzoo',
              'galaxyzoo.processing',
              'galaxyzoo.ui']
)