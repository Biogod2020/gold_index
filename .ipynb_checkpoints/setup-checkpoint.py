from setuptools import setup, find_packages

setup(
    name='cellbin_moran',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scanpy',
        # other dependencies
    ],
)
