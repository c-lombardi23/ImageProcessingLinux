from setuptools import setup, find_packages

setup(
    name='FiberCleaveProcessing',
    version='0.1.0',
    description='Fiber cleave quality classifier and tension predictor using CNN + MLP models',
    author='Chris Lombardi',
    author_email='clombardi23245@gmail.com',
    url='https://github.com/c-lombardi23/ImageProcessing',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'tensorflow==2.19.0',
        'keras-tuner==1.4.7',
        'numpy',
        'pandas',
        'scikit-learn',
        'joblib',
        'Pillow',
        'matplotlib',
        'pydantic'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    include_package_data=True,
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'cleave-app = cleave_app.main:main', 
        ],
    },
)
