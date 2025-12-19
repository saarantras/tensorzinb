try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open('README.md') as f:
    readme = f.read()

import sys
tensorflow = 'tensorflow_macos==2.15.0' if sys.platform == 'darwin' else 'tensorflow==2.15.0'

setup(
    name='tensorzinb',
    version='0.0.1',
    description='Zero Inflated Negative Binomial Model for Single-cell RNA-Sequencing Analysis using TensorFlow',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Tao Cui',
    author_email='taocui.caltech@gmail.com',
    url='https://github.com/wanglab/tensorzinb',
    keywords='Zero Inflated Negative Binomial scRNA-seq',
    packages=['tensorzinb'],
    include_package_data=True,
    install_requires=[
        'keras==2.15.0',
        'numpy>=1.26,<2',
        'pandas',
        'patsy',
        'scikit_learn',
        'scipy',
        'statsmodels',
        tensorflow,
    ],
    license='Apache',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.9,<3.12'
)
