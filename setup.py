try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open('README.md') as f:
    readme = f.read()

import sys
tensorflow = 'tensorflow_macos==2.16.1' if sys.platform == 'darwin' else 'tensorflow==2.16.1'

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
        'keras>=3.2,<3.5',
        'numpy>=1.26.4,<2',
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
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    python_requires='>=3.10,<3.13'
)
