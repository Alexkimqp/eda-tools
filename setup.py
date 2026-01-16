"""
Setup configuration for eda-tools package.
"""

from setuptools import setup
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
try:
    long_description = (this_directory / "README.md").read_text(encoding='utf-8')
except FileNotFoundError:
    long_description = "A collection of functions for exploratory data analysis"

setup(
    name='eda-tools',
    version='0.1.2',
    author='Kim Alexandr',
    author_email='alexkimaksashka@gmail.com',
    description='A collection of functions for exploratory data analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Alexkimqp/eda-tools',
    py_modules=['my_eda'],  # Указываем модуль напрямую, так как нет пакета
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.7',
    install_requires=[
        'pandas>=1.0.0',
        'numpy>=1.18.0',
        'matplotlib>=3.0.0',
        'seaborn>=0.10.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'flake8>=3.8',
        ],
    },
)

