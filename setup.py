from setuptools import setup, find_packages

setup(
    name='pygaello-vector',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # List your package dependencies here
    ],
    description='A set of custom packages used accross this code base',
    author='Cephas Soga',
    author_email='sogacephas@gmail.com',
    url='https://github.com/CephasSoga/Gaello-with-Janine.git',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)

