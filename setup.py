'''setup.py'''

from setuptools import setup, find_packages


with open("requirements.txt") as reqs_file:
    REQS = [line.rstrip() for line in reqs_file.readlines() if line[0] not in ['\n', '-', '#']]

setup(
    name =  'netdynflow',
    description = 'A package to study complex networks based on temporal flow propagations.',
    url =  'https://github.com/gorkazl/NetDynFlow',
    version =  '1.0.2',
    license =  'Apache License 2.0',

    author =  'Gorka Zamora-Lopez, Matthieu Gilson, Nikos Kouvaris',
    author_email =  'galib@zamora-lopez.xyz',

    install_requires =  REQS,
    packages =  find_packages(exclude=['doc', '*tests*']),
    scripts =  [],
    include_package_data =  True,

    keywords =  'graph theory, complex networks, network analysis, weighted networks',
    classifiers =  [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ]
)
