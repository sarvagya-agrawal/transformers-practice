from setuptools import setup
from glob import glob

setup(
    name='tod',
    use_scm_version=True,
    packages=[''],
    package_dir={'': 'src'},
    include_package_data=True,
    zip_safe=False,
    url='https://github.com/MathieuTuli/',
    python_requires='~=3.8',
    install_requires=[
    ],
    extras_require={
    },
    dependency_links=[
    ],
    setup_requires=[
    ],
    # DO NOT do tests_require; just call pytest or python -m pytest.
    license='License :: Other/Proprietary License',
    author='Mathieu Tuli',
    author_email='tuli.mathieu@gmail.com',
    description='Example of a production code package',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: Other/Proprietary License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
    ],
    scripts=glob('bin/*'),
)
