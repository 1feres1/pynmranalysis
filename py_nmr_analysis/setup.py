from setuptools import find_packages, setup
setup(
    name='pynmranalysis',
    packages=find_packages(include=['pynmranalysis']) ,
    version='0.5.1',
    description='python library for nmr quantification and analysis',
    author='Feres',
    license='MIT',
    install_requires=['numpy' , 'pandas ' ,"scipy " , "pyod == 0.9.0"  ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==6.2.4'],
    test_suite='tests',
)