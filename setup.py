import setuptools
import numpy

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="FLOPART",
    version="0.0.1",
    author="Tristan Miller",
    author_email="Tristan.Miller@nau.edu",
    description="A Python binding for FLOPART",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/deltarod/FLOPART/",
    install_requires=['numpy', 'pandas'],
    packages=['FLOPART'],
    extras_require={
        'test': ['pytest']
    },
    ext_modules=[setuptools.Extension('FLOPARTInterface',
                                      ['src/interface.cpp', 'src/FLOPART.cpp', 'src/funPieceListLog.cpp'],
                                      include_dirs=[numpy.get_include()],
                                      extra_compile_args=['-std=c++11'])],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: C",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires='>=3.6',
)
