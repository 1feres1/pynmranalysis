from setuptools import find_packages, setup ,Extension


with open('E:\my work\py_nmr_analysis\Readme.md', "r", encoding="utf-8") as fh:
    long_description = fh.read()



setup(
    name='pynmranalysis',
    packages=find_packages(include=['pynmranalysis']) ,
    version='1.1.3',
    description='python library for NMR preprocessing and analysis',
    long_description = long_description ,
    long_description_content_type='text/markdown',
    url="https://github.com/1feres1/pynmranalysis/",
    author='Feres Sakouhi',
    author_email='feressakouhi@gmail.com',
    license='MIT',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux'] ,
    install_requires=['numpy ' , 'pandas==1.2.4' ,"scipy" ,"scikit-learn" ,"matplotlib"
                      ],

    setup_requires=['pytest-runner'],
    tests_require=['pytest==6.2.4'],
    test_suite='tests',
)